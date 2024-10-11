from fastapi import FastAPI, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from youtube_utils import YouTubeService
from whisper_transcription import WhisperTranscriptionService
from langchain_utils import LangChainService
from config import settings

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


def get_youtube_service(settings: settings = Depends(get_settings)):
    return YouTubeService(settings)


def get_whisper_service(settings: settings = Depends(get_settings)):
    return WhisperTranscriptionService(settings)


def get_langchain_service(settings: settings = Depends(get_settings)):
    return LangChainService(settings)


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
    await langchain_service.prepare_retriever(transcript)
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
    data = await request.json()
    prompt = data.get("prompt")

    async def generate():
        async for chunk in langchain_service.stream_chat(prompt):
            yield f"data: {chunk}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
