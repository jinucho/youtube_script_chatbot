import runpod
import asyncio
from fastapi import FastAPI, Request, Depends, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse, Response
from langchain_utils import LangChainService
from whisper_transcription import WhisperTranscriptionService
from youtube_utils import YouTubeService
import json
import logging
import os
import multiprocessing

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"  # OpenMP 스레드 수 제한

# 멀티프로세싱 설정 조정
multiprocessing.set_start_method("spawn", force=True)

app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 로그 설정
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


# 의존성 주입 설정
def get_youtube_service():
    return YouTubeService()


def get_whisper_service():
    return WhisperTranscriptionService()


def get_session_id(request: Request) -> str:
    session_id = request.headers.get("x-session-id")
    if not session_id:
        raise HTTPException(status_code=400, detail="Session ID is required in headers")
    logger.debug(f"get_session_id called with session_id: {session_id}")
    return session_id


def get_langchain_service(session_id: str = Depends(get_session_id)):
    return LangChainService.get_instance(session_id)


# 요청 및 응답 로깅 미들웨어 추가
@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.debug(f"Request URL: {request.url}")
    logger.debug(f"Request method: {request.method}")
    logger.debug(f"Request headers: {dict(request.headers)}")
    try:
        body = await request.json()
        logger.debug(f"Request body: {body}")
    except Exception:
        logger.debug("Request body is not JSON.")
    response: Response = await call_next(request)
    response_body = b""
    async for chunk in response.body_iterator:
        response_body += chunk
    logger.debug(f"Response status code: {response.status_code}")
    logger.debug(f"Response body: {response_body.decode('utf-8')}")
    response = Response(
        content=response_body,
        status_code=response.status_code,
        headers=dict(response.headers),
    )
    return response


@app.get("/get_title_hash")
async def get_title_hash(
    url: str = Query(..., description="YouTube URL을 입력하세요."),
    youtube_service: YouTubeService = Depends(get_youtube_service),
):
    if not url or not isinstance(url, str):
        raise HTTPException(status_code=422, detail="Invalid URL format or missing URL")
    logger.debug(f"Received URL in get_title_hash: {url}")
    try:
        title_and_hashtags = await youtube_service.get_title_and_hashtags(url)
        logger.debug(f"Title and Hashtags for URL {url}: {title_and_hashtags}")
        return title_and_hashtags
    except Exception as e:
        logger.error(f"Error fetching title and hashtags for URL {url}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to fetch title and hashtags: {e}"
        )


@app.get("/get_script_summary")
async def get_script_summary(
    url: str = Query(..., description="YouTube URL을 입력하세요."),
    session_id: str = Depends(get_session_id),
    youtube_service: YouTubeService = Depends(get_youtube_service),
    whisper_service: WhisperTranscriptionService = Depends(get_whisper_service),
    langchain_service: LangChainService = Depends(get_langchain_service),
):
    if not url or not isinstance(url, str):
        logger.error(f"Invalid URL parameter in get_script_summary: {url}")
        raise HTTPException(status_code=422, detail="Invalid URL format or missing URL")
    logger.debug(f"Received URL in get_script_summary: {url}")
    logger.debug(f"Session ID in get_script_summary: {session_id}")
    try:
        video_info = await youtube_service.get_video_info(url)
        logger.debug(f"[Session {session_id}] Video info for URL {url}: {video_info}")
        transcript = await whisper_service.transcribe(video_info["audio_url"])
        logger.debug(
            f"[Session {session_id}] Transcript for video {video_info['audio_url']}: {transcript}"
        )
        summary = await langchain_service.summarize(transcript)
        logger.debug(f"[Session {session_id}] Summary for transcript: {summary}")
        return {
            "summary_result": summary,
            "language": transcript["language"],
            "script": transcript["script"],
        }
    except KeyError as e:
        logger.error(
            f"[Session {session_id}] KeyError: Missing expected key in response data - {e}"
        )
        raise HTTPException(status_code=400, detail=f"Invalid data structure: {e}")
    except HTTPException as e:
        logger.error(f"[Session {session_id}] HTTPException: {e.detail}")
        raise e
    except Exception as e:
        logger.error(
            f"[Session {session_id}] Unhandled exception in get_script_summary: {e}"
        )
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rag_stream_chat")
async def rag_stream_chat(
    request: Request,
    session_id: str = Depends(get_session_id),
    langchain_service: LangChainService = Depends(get_langchain_service),
):
    if not session_id:
        logger.error("Session ID is missing in request headers for rag_stream_chat.")
        raise HTTPException(
            status_code=400, detail="Session ID is required in request headers"
        )
    try:
        data = await request.json()
        prompt = data.get("prompt")
        if not prompt:
            logger.error(f"[Session {session_id}] Prompt is missing in request body.")
            raise HTTPException(status_code=400, detail="Prompt is required")

        async def generate():
            try:
                async for chunk in langchain_service.stream_chat(prompt):
                    yield f"data: {json.dumps({'content': chunk})}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as e:
                logger.error(f"[Session {session_id}] Error during stream_chat: {e}")
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
                yield "data: [DONE]\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")
    except json.JSONDecodeError:
        logger.error(f"[Session {session_id}] Invalid JSON in request body.")
        return JSONResponse(
            status_code=400, content={"error": "Invalid JSON in request body"}
        )
    except Exception as e:
        logger.error(
            f"[Session {session_id}] Unhandled exception in rag_stream_chat: {e}"
        )
        return JSONResponse(status_code=500, content={"error": str(e)})


# RunPod handler 통합
def runpod_handler(event):
    async def async_handler():
        print(f"Received event: {event}")
        endpoint = event["input"].get("endpoint", "/")
        method = event["input"].get("method", "POST").upper()
        request_data = event["input"].get("params", {})
        headers = event["input"].get("headers", {})
        session_id = headers.get("x-session-id")

        if method == "POST":
            if endpoint == "/rag_stream_chat":
                request = Request(
                    scope={
                        "type": "http",
                        "method": method,
                        "path": endpoint,
                        "headers": [
                            (k.encode(), v.encode()) for k, v in headers.items()
                        ],
                    },
                    receive=None,
                )
                request._json = request_data
                response = await rag_stream_chat(request)
                return response
            else:
                return {"error": "Invalid endpoint or method"}
        elif method == "GET":
            if endpoint == "/get_title_hash":
                response = await get_title_hash(url=request_data.get("url"))
                return response
            elif endpoint == "/get_script_summary":
                response = await get_script_summary(
                    url=request_data.get("url"), session_id=session_id
                )
                return response
            else:
                return {"error": "Invalid endpoint or method"}

    if asyncio.get_event_loop().is_running():
        return asyncio.create_task(async_handler())
    else:
        return asyncio.run(async_handler())


if __name__ == "__main__":
    runpod.serverless.start({"handler": runpod_handler})
