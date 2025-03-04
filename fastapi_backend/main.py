import asyncio
import logging
import os
import stat
import warnings

from fastapi import FastAPI, Depends, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from config import settings
from youtube_utils import YouTubeService
from whisper_transcription import WhisperTranscriptionService
from langchain_utils import LangChainService

# 로그 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

warnings.filterwarnings(action="ignore")

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"  # OpenMP 스레드 수 제한

# 전역 인스턴스 저장
youtube_service_instance = None
whisper_service_instance = None
langchain_service_cache = {}

DATA_PATH = settings.DATA_PATH

# FastAPI 앱 생성
app = FastAPI(
    title="YouTube Script Chatbot API",
    version="1.0.0",
    description="YouTube 비디오 스크립트 추출 및 챗봇 API",
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 실제 배포 시 특정 도메인으로 제한하는 것이 좋습니다
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 요청 모델 정의
class TitleHashRequest(BaseModel):
    url: str = Field(..., description="YouTube 비디오 URL")
    url_id: str = Field(..., description="고유 URL ID")


class ScriptSummaryRequest(BaseModel):
    url: str = Field(..., description="YouTube 비디오 URL")
    url_id: str = Field(..., description="고유 URL ID")


class ChatRequest(BaseModel):
    prompt: str = Field(..., description="사용자 질문")
    url_id: str = Field(..., description="고유 URL ID")


def setup_volume():
    """데이터 저장을 위한 볼륨 디렉토리를 설정합니다."""
    try:
        # data 디렉토리가 없으면 생성
        os.makedirs(DATA_PATH, exist_ok=True)

        # 권한 설정
        os.chmod(
            DATA_PATH, stat.S_IRWXU | stat.S_IRWXG | stat.S_IROTH | stat.S_IXOTH
        )  # 755

        logger.info(f"Successfully set up volume directories and permissions")
    except Exception as e:
        logger.error(f"Error setting up volume: {e}")
        raise


# 서비스 인스턴스를 비동기적으로 관리하는 함수
async def get_service_instances(session_id=None, url_id=None):
    """
    서비스 인스턴스를 관리하는 함수입니다.

    Args:
        session_id (str, optional): 세션 ID
        url_id (str, optional): URL ID

    Returns:
        tuple: (YouTubeService, WhisperTranscriptionService, LangChainService) 인스턴스
    """
    global youtube_service_instance, whisper_service_instance

    # YouTubeService 인스턴스 생성 (싱글톤)
    if youtube_service_instance is None:
        youtube_service_instance = YouTubeService()

    # WhisperTranscriptionService 인스턴스 생성 (싱글톤)
    if whisper_service_instance is None:
        whisper_service_instance = WhisperTranscriptionService(url_id)

    # LangChainService 인스턴스 생성 (세션별)
    if session_id:
        if session_id not in langchain_service_cache:
            langchain_service_cache[session_id] = LangChainService.get_instance(
                session_id
            )
        langchain_service = langchain_service_cache[session_id]
    else:
        langchain_service = None

    return youtube_service_instance, whisper_service_instance, langchain_service


# 의존성 주입을 위한 함수
async def get_session_id(x_session_id: str = Header(None)):
    """
    요청 헤더에서 세션 ID를 추출합니다.

    Args:
        x_session_id (str): 요청 헤더의 X-Session-ID 값

    Returns:
        str: 세션 ID

    Raises:
        HTTPException: 세션 ID가 없는 경우 400 오류 발생
    """
    if not x_session_id:
        raise HTTPException(status_code=400, detail="X-Session-ID header is required")
    return x_session_id


# 비동기 작업 함수들
async def get_title_hash(url: str, url_id: str, youtube_service: YouTubeService):
    """
    YouTube 비디오의 제목과 해시태그를 가져옵니다.

    Args:
        url (str): YouTube 비디오 URL
        url_id (str): 고유 URL ID
        youtube_service (YouTubeService): YouTubeService 인스턴스

    Returns:
        str: 제목과 해시태그 문자열

    Raises:
        ValueError: URL이 유효하지 않거나 데이터를 가져오는 데 실패한 경우
    """
    if not url or not isinstance(url, str):
        raise ValueError("Invalid URL format or missing URL")

    logger.info(f"Received URL in get_title_hash: {url}")
    try:
        video_data = await youtube_service.get_video_data(url, url_id)
        title_and_hashtags = video_data.get("title_hashtags", "")
        logger.info(f"Title and Hashtags for URL {url}: {title_and_hashtags}")
        return title_and_hashtags
    except Exception as e:
        logger.error(f"Error fetching title and hashtags for URL {url}: {e}")
        raise ValueError(f"Failed to fetch title and hashtags: {e}")


async def get_script_summary(
    url: str,
    session_id: str,
    url_id: str,
    youtube_service: YouTubeService,
    whisper_service: WhisperTranscriptionService,
    langchain_service: LangChainService,
):
    """
    YouTube 비디오의 스크립트를 추출하고 요약합니다.

    Args:
        url (str): YouTube 비디오 URL
        session_id (str): 세션 ID
        url_id (str): 고유 URL ID
        youtube_service (YouTubeService): YouTubeService 인스턴스
        whisper_service (WhisperTranscriptionService): WhisperTranscriptionService 인스턴스
        langchain_service (LangChainService): LangChainService 인스턴스

    Returns:
        dict: 요약 결과, 추천 질문, 언어, 스크립트를 포함한 딕셔너리

    Raises:
        ValueError: URL이 유효하지 않거나 처리 중 오류가 발생한 경우
    """
    if not url or not isinstance(url, str):
        raise ValueError("Invalid URL format or missing URL")

    logger.info(f"Received URL in get_script_summary: {url}")
    logger.info(f"Session ID in get_script_summary: {session_id}")
    try:
        video_info = await youtube_service.get_video_data(url=url, url_id=url_id)
        title_hash = video_info.get("title_hashtags", "")
        audio_url = video_info.get("audio_url", "")
        logger.info(f"[Session {session_id}] Video info for URL {url}: {video_info}")
        transcript = await whisper_service.transcribe(audio_url, title_hash, url_id)
        logger.info(
            f"[Session {session_id}] Transcript for video {video_info['audio_url'][:10]}: {transcript.get('script')[:3]}"
        )

        summary, questions = await langchain_service.summarize(transcript, url_id)
        logger.info(f"[Session {session_id}] Summary for transcript: {summary[:10]}")

        return {
            "summary_result": summary,
            "recommended_questions": questions,
            "language": transcript["language"],
            "script": transcript["script"],
        }
    except KeyError as e:
        logger.error(
            f"[Session {session_id}] KeyError: Missing expected key in response data - {e}"
        )
        raise ValueError(f"Invalid data structure: {e}")
    except Exception as e:
        logger.error(
            f"[Session {session_id}] Unhandled exception in get_script_summary: {e}"
        )
        raise ValueError(str(e))


async def rag_stream_chat(
    prompt: str, session_id: str, url_id: str, langchain_service: LangChainService
):
    """
    RAG 기반 스트리밍 채팅 응답을 생성합니다.

    Args:
        prompt (str): 사용자 질문
        session_id (str): 세션 ID
        url_id (str): 고유 URL ID
        langchain_service (LangChainService): LangChainService 인스턴스

    Returns:
        async generator: 스트리밍 응답을 위한 비동기 제너레이터

    Raises:
        ValueError: 세션 ID나 프롬프트가 없는 경우
    """
    if not session_id:
        raise ValueError("Session ID is required")

    if not prompt:
        raise ValueError("Prompt is required")

    async def generate():
        try:
            async for chunk in langchain_service.stream_chat(prompt, url_id):
                yield f"data: {chunk}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            logger.error(f"[Session {session_id}] Error during stream_chat: {e}")
            yield f'data: {{"error": "{str(e)}"}}\n\n'
            yield "data: [DONE]\n\n"

    return generate()


# 라우트 정의
@app.get("/", tags=["root"])
async def root():
    """API 루트 경로"""
    return {"message": "YouTube Script Chatbot API"}


@app.post("/api/title-hash", tags=["youtube"])
async def title_hash_endpoint(
    request: TitleHashRequest, session_id: str = Depends(get_session_id)
):
    """
    YouTube 비디오의 제목과 해시태그를 가져옵니다.

    Args:
        request (TitleHashRequest): 요청 모델
        session_id (str): 세션 ID

    Returns:
        dict: 제목과 해시태그 정보
    """
    try:
        youtube_service, _, _ = await get_service_instances(session_id)
        result = await get_title_hash(
            url=request.url, url_id=request.url_id, youtube_service=youtube_service
        )
        return {"title_hash": result}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error in title_hash_endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/script-summary", tags=["script"])
async def script_summary_endpoint(
    request: ScriptSummaryRequest, session_id: str = Depends(get_session_id)
):
    """
    YouTube 비디오의 스크립트를 추출하고 요약합니다.

    Args:
        request (ScriptSummaryRequest): 요청 모델
        session_id (str): 세션 ID

    Returns:
        dict: 요약 결과, 추천 질문, 언어, 스크립트 정보
    """
    try:
        youtube_service, whisper_service, langchain_service = (
            await get_service_instances(session_id, request.url_id)
        )
        result = await get_script_summary(
            url=request.url,
            session_id=session_id,
            url_id=request.url_id,
            youtube_service=youtube_service,
            whisper_service=whisper_service,
            langchain_service=langchain_service,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error in script_summary_endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat", tags=["chat"])
async def chat_endpoint(
    request: ChatRequest, session_id: str = Depends(get_session_id)
):
    """
    RAG 기반 스트리밍 채팅 응답을 생성합니다.

    Args:
        request (ChatRequest): 요청 모델
        session_id (str): 세션 ID

    Returns:
        StreamingResponse: 스트리밍 응답
    """
    try:
        _, _, langchain_service = await get_service_instances(
            session_id, request.url_id
        )

        stream = await rag_stream_chat(
            prompt=request.prompt,
            session_id=session_id,
            url_id=request.url_id,
            langchain_service=langchain_service,
        )

        return StreamingResponse(stream, media_type="text/event-stream")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error in chat_endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# 애플리케이션 시작 시 볼륨 설정
@app.on_event("startup")
async def startup_event():
    """애플리케이션 시작 시 실행되는 이벤트 핸들러"""
    setup_volume()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
