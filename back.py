import os
import re
import tempfile  # 임시 파일 경로 생성을 위한 tempfile 라이브러리
import uuid  # 고유한 파일 이름 생성을 위한 uuid 라이브러리

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel
from pydantic import BaseModel
from pytubefix import YouTube
from pytubefix.cli import on_progress
from youtube_transcript_api import VideoUnavailable
import time

# 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Whisper 모델 로드
model = WhisperModel("large-v3", device="cuda", compute_type="float16")

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


async def get_title_hash(url):
    try:
        # YouTube 객체 생성
        yt = YouTube(url)

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
        # # 유튜브 영상 ID 추출
        # video_id = url.split("=")[1]

        # # 1. 자막 추출 시도
        # try:
        #     transcript = YouTubeTranscriptApi.get_transcript(
        #         video_id, languages=["ko", "en"]
        #     )
        #     return transcript
        # except (NoTranscriptFound, TranscriptsDisabled):
        #     print("자막이 없거나 비활성화되어 있습니다. 음성 추출을 시도합니다.")
        # 실제 내용에 연관된 자막이 아닌 경우가 있어서 제외

        # 2. 자막이 없는 경우, 음성 추출 및 Whisper 사용
        # 유튜브 영상에서 오디오 스트림 다운로드
        start_time = time.time()
        yt = YouTube(url, on_progress_callback=on_progress)
        audio_stream = yt.streams.filter(only_audio=True).first()

        # 임시 디렉터리 및 파일 이름 생성
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_audio_file = os.path.join(temp_dir, f"{uuid.uuid4()}_temp_audio.wav")

            # 오디오 데이터를 임시 파일로 저장
            audio_stream.download(output_path=temp_dir, filename=temp_audio_file)
            print(f"오디오 스트림 다운로드 완료: {time.time() - start_time}초 소요")
            # 오디오 파일을 Whisper 모델에 입력하여 텍스트 변환 수행
            start_time = time.time()
            segments, info = model.transcribe(
                temp_audio_file,
                beam_size=5,
                task="transcribe",
                language="ko",
                condition_on_previous_text=False,
            )

            # Whisper 결과에서 각 세그먼트를 사용하여 시간 정보와 텍스트 추출
            checked_text = {}
            transcript_with_timestamps = []

            for segment in segments:
                start = segment.start  # 시작 시간 (초 단위)
                end = segment.end  # 종료 시간 (초 단위)
                text = segment.text.strip()  # 텍스트

                # 이미 확인된 텍스트가 없는 경우 새로 추가하고, end 시간도 기록 --> 일부 텍스트가 중복 되는 경우가 있어 제외 처리
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
                    # 기존 end 시간을 업데이트하여 가장 늦은 종료 시간을 반영
                    for item in transcript_with_timestamps:
                        if item["text"] == text:
                            item["end"] = round(end, 2)
                            break
                        # 3. 임시 파일 삭제
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
