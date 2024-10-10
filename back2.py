import asyncio
import json
import os
import re
import tempfile  # 임시 파일 경로 생성을 위한 tempfile 라이브러리
import time
import uuid  # 고유한 파일 이름 생성을 위한 uuid 라이브러리
import warnings
from concurrent.futures import ThreadPoolExecutor

import torch
import uvicorn
from deep_translator import GoogleTranslator
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import PromptTemplate
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.document_loaders import JSONLoader, TextLoader
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_teddynote import logging
from langchain_teddynote.callbacks import StreamingCallback
from pydantic import BaseModel
from pytubefix import YouTube
from pytubefix.cli import on_progress
from youtube_transcript_api import VideoUnavailable

tl = GoogleTranslator(source="en", target="ko")

warnings.filterwarnings(action="ignore")

load_dotenv()  # .env 파일에서 환경 변수를 로드합니다
api_key = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = "Youtube_Script_Chatbot"
logging.langsmith("Youtube_Script_Chatbot")

# 디바이스 설정
device = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "float16" if device == "cuda" else "int8"

summary_prompt = hub.pull("teddynote/summary-stuff-documents-korean")
summary_prompt.template = 'Please summarize the sentence according to the following REQUEST.\nREQUEST:\n1. Summarize the main points in bullet points in KOREAN.\n2. Each summarized sentence must start with an emoji that fits the meaning of the each sentence.\n3. Use various emojis to make the summary more interesting.\n4. Translate the summary into KOREAN if it is written in ENGLISH.\n5. DO NOT translate any technical terms.\n6. DO NOT include any unnecessary information.\n7. Please refer to each summary and indicate the key topic.\n\nCONTEXT:\n{context}\n\nSUMMARY:"\n'

llm = ChatOpenAI(
    model_name="gpt-4o-mini",
    streaming=True,
    temperature=0.7,
    callbacks=[StreamingCallback()],
)

# Whisper 모델 로드
model = WhisperModel("large-v3", device=device, compute_type=compute_type)

# FastAPI 애플리케이션 생성
app = FastAPI()

# CORS 설정 (Streamlit과의 통신 허용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class YouTubeRequest(BaseModel):
    url: str


# 비동기 작업 실행을 위한 헬퍼 함수 정의
async def run_in_threadpool(func, *args, **kwargs):
    loop = asyncio.get_event_loop()
    # `lambda`를 사용하여 키워드 인수(`kwargs`)를 전달할 수 있는 함수 생성
    partial_func = lambda: func(*args, **kwargs)
    # `run_in_executor`를 사용하여 스레드 풀에서 비동기 작업 실행
    return await loop.run_in_executor(None, partial_func)


# 비동기적으로 YouTube 객체 생성
async def create_youtube_instance(url):
    # YouTube 객체를 생성하면서 on_progress_callback을 전달
    return await run_in_threadpool(YouTube, url, on_progress_callback=on_progress)


@app.get("/get_title_hash")
async def get_title_hash(url):
    try:
        # 비동기적으로 YouTube 객체 생성
        yt = await create_youtube_instance(url)

        # 1. 영상 제목 추출
        title = yt.title

        # 2. 해시태그 추출
        description = yt.description
        hashtags = re.findall(r"#\w+", description)
        hashtags = " ".join(hashtags)

        return {"title": title, "hashtags": hashtags}
    except Exception as e:
        return f"Error: {str(e)}"


async def get_script(url):
    try:
        start_time = time.time()

        # 비동기적으로 YouTube 객체 생성
        yt = await create_youtube_instance(url)
        # 오디오 스트림 필터링
        audio_stream = yt.streams.filter(only_audio=True).first()

        if not audio_stream:
            raise HTTPException(
                status_code=404, detail="오디오 스트림을 찾을 수 없습니다."
            )

        # 임시 디렉터리 및 파일 이름 생성
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_audio_file = os.path.join(temp_dir, f"{uuid.uuid4()}_temp_audio.wav")

            # 오디오 데이터를 임시 파일로 저장
            await run_in_threadpool(
                audio_stream.download, output_path=temp_dir, filename=temp_audio_file
            )
            print(f"오디오 스트림 다운로드 완료: {time.time() - start_time}초 소요")

            # 오디오 파일을 Whisper 모델에 입력하여 텍스트 변환 수행
            start_time = time.time()
            segments, info = model.transcribe(
                temp_audio_file,
                beam_size=5,
                task="transcribe",
                # language=language,
                condition_on_previous_text=False,
            )
            language = ""
            if info.language == "en":
                language = "en"
            elif info.language == "ko":
                language = "ko"
            # Whisper 결과에서 각 세그먼트를 사용하여 시간 정보와 텍스트 추출
            checked_text = {}
            transcript_with_timestamps = []

            for segment in segments:
                start = segment.start  # 시작 시간 (초 단위)
                end = segment.end  # 종료 시간 (초 단위)
                text = segment.text.strip()  # 텍스트

                # 이미 확인된 텍스트가 없는 경우 새로 추가하고, end 시간도 기록
                if text not in checked_text:
                    checked_text[text] = end  # text의 마지막 end 시간을 기록
                    transcript_with_timestamps.append(
                        {
                            "text": text,
                            "start": round(start, 2),
                            "end": round(end, 2),  # 처음 등장한 end 시간
                        }
                    )
                else:
                    # 이미 존재하는 텍스트의 경우 마지막 end 시간 업데이트
                    for item in transcript_with_timestamps:
                        if item["text"] == text:
                            item["end"] = round(end, 2)
                            break

            # 임시 파일 삭제
            if os.path.exists(temp_audio_file):
                os.remove(temp_audio_file)

        # 변환된 텍스트 반환
        print(f"텍스트 변환 완료: {time.time() - start_time}초 소요")
        return {"script": transcript_with_timestamps, "language": language}

    except VideoUnavailable:
        return "Error: 이 영상을 찾을 수 없습니다. 링크가 올바른지 확인하세요."
    except Exception as e:
        return f"Error: 알 수 없는 오류가 발생했습니다. {str(e)}"


@app.get("/get_script_summary")
async def get_script_summary(url):
    try:
        global SCRIPT
        SCRIPT = await get_script(url)
        script = SCRIPT.copy()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_script = os.path.join(temp_dir, f"{uuid.uuid4()}_temp_script.json")
            with open(temp_script, "w", encoding="utf-8") as f:
                json.dump(SCRIPT["script"], f, ensure_ascii=False, indent=4)
            jq_schema = ".[].text"
            docs = JSONLoader(
                file_path=temp_script, jq_schema=jq_schema, text_content=False
            ).load()
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_script = os.path.join(temp_dir, f"{uuid.uuid4()}_temp_script.txt")
                with open(temp_script, "w") as file:
                    for text in [doc.page_content for doc in docs]:
                        file.write(text + "\n")
                docs = TextLoader(temp_script).load()
                stuff_chain = create_stuff_documents_chain(llm, summary_prompt)
                summary_result = stuff_chain.invoke({"context": docs})
                if os.path.exists(temp_script):
                    os.remove(temp_script)
                return {
                    "summary_result": summary_result,
                    "language": SCRIPT["language"],
                }
    except Exception as e:
        return f"Error: {str(e)}"


@app.get("/get_script")
async def get_script_summary(language):
    try:
        scripts = SCRIPT["script"].copy()
        if language == "ko":
            return {"script": scripts}
        elif language == "en":
            for script in scripts:
                script["text"] = tl.translate(script["text"], source="en", target="ko")
            return {"script": scripts}

    except Exception as e:
        return f"Error: {str(e)}"


@app.get("/chat")
async def chat(prompt: str):
    result = llm.invoke(prompt)
    return {"result": result.content}


# FastAPI 서버 실행
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8010)
