import asyncio
import os
import re
import tempfile  # 임시 파일 경로 생성을 위한 tempfile 라이브러리
import time
import uuid  # 고유한 파일 이름 생성을 위한 uuid 라이브러리
from concurrent.futures import ThreadPoolExecutor

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel
from pydantic import BaseModel
from pytubefix import YouTube
from pytubefix.cli import on_progress
from youtube_transcript_api import VideoUnavailable

# 디바이스 설정
device = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "float16" if device=="cuda" else "int8"

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


async def get_script(url, language="ko"):
    try:
        start_time = time.time()

        # 비동기적으로 YouTube 객체 생성
        yt = await create_youtube_instance(url)
        # 오디오 스트림 필터링
        audio_stream = yt.streams.filter(only_audio=True).first()
        
        if not audio_stream:
            raise HTTPException(status_code=404, detail="오디오 스트림을 찾을 수 없습니다.")

        # 임시 디렉터리 및 파일 이름 생성
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_audio_file = os.path.join(temp_dir, f"{uuid.uuid4()}_temp_audio.wav")

            # 오디오 데이터를 임시 파일로 저장
            await run_in_threadpool(audio_stream.download, output_path=temp_dir, filename=temp_audio_file)
            print(f"오디오 스트림 다운로드 완료: {time.time() - start_time}초 소요")

            # 오디오 파일을 Whisper 모델에 입력하여 텍스트 변환 수행
            start_time = time.time()
            segments, info = model.transcribe(
                temp_audio_file,
                beam_size=5,
                task="transcribe",
                language=language,
                condition_on_previous_text=False,
            )

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
        return transcript_with_timestamps

    except VideoUnavailable:
        return "Error: 이 영상을 찾을 수 없습니다. 링크가 올바른지 확인하세요."
    except Exception as e:
        return f"Error: 알 수 없는 오류가 발생했습니다. {str(e)}"


@app.get("/extract_info")
async def extract_info(url: str):
    try:
        # 유튜브 URL을 입력받아 정보 추출
        print(f"URL: {url}")
        title_hash = await get_title_hash(url)
        script = await get_script(url)
        return {"title_hash": title_hash, "script": script}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"정보를 추출하는 중 오류가 발생했습니다: {str(e)}"
        )


# FastAPI 서버 실행
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8010)
