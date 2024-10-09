import asyncio
import os
import re
import tempfile
import time
import uuid
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
compute_type = "float16" if device == "cuda" else "int8"

# 모델 로드
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


# 비동기 작업 큐 생성 (동시에 최대 2개의 작업만 처리하도록 제한)
task_queue = asyncio.Queue()
max_concurrent_tasks = 2  # 동시에 처리 가능한 최대 작업 수
semaphore = asyncio.Semaphore(max_concurrent_tasks)


async def create_youtube_instance(url):
    loop = asyncio.get_event_loop()
    yt = await loop.run_in_executor(
        None, lambda: YouTube(url, on_progress_callback=on_progress)
    )
    return yt


async def get_title_hash(url):
    try:
        yt = await create_youtube_instance(url)
        title = yt.title
        description = yt.description
        hashtags = re.findall(r"#\w+", description)
        hashtags = " ".join(hashtags)
        return {"title": title, "hashtags": hashtags}
    except Exception as e:
        return f"Error: {str(e)}"


async def process_script_task(url, language="ko"):
    loop = asyncio.get_event_loop()
    try:
        async with semaphore:  # 세마포어를 사용하여 동시 작업 수를 제한
            start_time = time.time()
            yt = await create_youtube_instance(url)
            audio_stream = yt.streams.filter(only_audio=True).first()

            if not audio_stream:
                raise HTTPException(
                    status_code=404, detail="오디오 스트림을 찾을 수 없습니다."
                )

            with tempfile.TemporaryDirectory() as temp_dir:
                temp_audio_file = os.path.join(
                    temp_dir, f"{uuid.uuid4()}_temp_audio.wav"
                )

                # 오디오 다운로드를 비동기로 처리
                await asyncio.get_event_loop().run_in_executor(
                    None, audio_stream.download, temp_dir, temp_audio_file
                )
                print(f"오디오 스트림 다운로드 완료: {time.time() - start_time}초 소요")

                # Whisper 모델을 동기적으로 호출 (GPU 처리에서 비동기 호출의 이점이 없음)
                # segments, info = model.transcribe(
                #     temp_audio_file, beam_size=5, task="transcribe", language=language
                # )
                segments, info = await loop.run_in_executor(
                    None,
                    lambda: model.transcribe(
                        temp_audio_file,
                        beam_size=5,
                        task="transcribe",
                        language=language,
                    ),
                )

                transcript_with_timestamps = [
                    {
                        "text": segment.text.strip(),
                        "start": round(segment.start, 2),
                        "end": round(segment.end, 2),
                    }
                    for segment in segments
                ]

            print(f"텍스트 변환 완료: {time.time() - start_time}초 소요")
            return transcript_with_timestamps

    except VideoUnavailable:
        return "Error: 이 영상을 찾을 수 없습니다. 링크가 올바른지 확인하세요."
    except Exception as e:
        return f"Error: 알 수 없는 오류가 발생했습니다. {str(e)}"


# 작업 큐에서 작업을 순차적으로 처리하는 백그라운드 작업자
async def worker():
    while True:
        url, language, result_future = (
            await task_queue.get()
        )  # 작업 큐에서 작업을 가져옴
        try:
            result = await process_script_task(url, language)
            result_future.set_result(result)
        except Exception as e:
            result_future.set_exception(e)
        finally:
            task_queue.task_done()  # 작업 완료 신호


@app.get("/extract_info")
async def extract_info(url: str, language: str = "ko"):
    try:
        title_hash = await get_title_hash(url)

        # 작업 결과를 기다리는 Future 객체 생성
        result_future = asyncio.get_event_loop().create_future()
        await task_queue.put((url, language, result_future))  # 작업 큐에 작업 추가
        script = await result_future  # 작업이 완료될 때까지 대기

        return {"title_hash": title_hash, "script": script}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"정보를 추출하는 중 오류가 발생했습니다: {str(e)}"
        )


# FastAPI 애플리케이션이 시작될 때 작업자 코루틴 실행
@app.on_event("startup")
async def startup_event():
    # 작업자 코루틴을 별도의 작업으로 실행하여 백그라운드에서 큐를 처리하도록 설정
    asyncio.create_task(worker())


# FastAPI 서버 실행
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8010)
