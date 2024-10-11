# main.py
import json

from config import settings
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from langchain_utils import LangChainService
from whisper_transcription import WhisperTranscriptionService
from youtube_utils import YouTubeService

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_settings():
    return settings


def get_youtube_service():
    return YouTubeService()


def get_whisper_service():
    return WhisperTranscriptionService()


def get_langchain_service():
    return LangChainService()


@app.get("/get_title_hash")
async def get_title_hash(
    url: str, youtube_service: YouTubeService = Depends(get_youtube_service)
):
    return await youtube_service.get_title_and_hashtags(url)


@app.get("/get_script_summary")
async def get_script_summary(
    url: str,
    youtube_service: YouTubeService = Depends(get_youtube_service),
    whisper_service: WhisperTranscriptionService = Depends(get_whisper_service),
    langchain_service: LangChainService = Depends(get_langchain_service),
):
    video_info = await youtube_service.get_video_info(url)
    transcript = await whisper_service.transcribe(video_info["audio_url"])
    summary = await langchain_service.summarize(transcript)
    # prepare_retriever는 summarize 메서드 내에서 자동으로 호출되므로 여기서는 제거합니다.
    return {
        "summary_result": summary,
        "language": transcript["language"],
        "script": transcript["script"],
    }


@app.post("/rag_stream_chat")
async def rag_stream_chat(
    request: Request,
    langchain_service: LangChainService = Depends(get_langchain_service),
):
    try:
        data = await request.json()
        prompt = data.get("prompt")
        if not prompt:
            raise HTTPException(status_code=400, detail="Prompt is required")

        async def generate():
            try:
                async for chunk in langchain_service.stream_chat(prompt):
                    yield f"data: {json.dumps({'content': chunk})}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
                yield "data: [DONE]\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")
    except json.JSONDecodeError:
        return JSONResponse(
            status_code=400, content={"error": "Invalid JSON in request body"}
        )
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
