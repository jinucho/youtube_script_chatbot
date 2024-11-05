import asyncio
import logging
import os
import warnings

import runpod
from config import backup_data
from langchain_utils import LangChainService
from whisper_transcription import WhisperTranscriptionService
from youtube_utils import YouTubeService

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

CURRENT_DIR = os.getcwd()

os.makedirs(os.path.join(CURRENT_DIR, "data"), exist_ok=True)


# 서비스 인스턴스를 비동기적으로 관리하는 함수
async def get_service_instances(session_id=None, url_id=None):
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


# 비동기 작업 함수들
async def get_title_hash(url: str, url_id: str, youtube_service: YouTubeService):
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
    if not session_id:
        raise ValueError("Session ID is required")

    if not prompt:
        raise ValueError("Prompt is required")

    async def generate():
        try:
            async for chunk in langchain_service.stream_chat(prompt, url_id):
                yield {"content": chunk}
            yield {"content": "[DONE]"}
        except Exception as e:
            logger.error(f"[Session {session_id}] Error during stream_chat: {e}")
            yield {"error": str(e)}
            yield {"content": "[DONE]"}

    return [chunk async for chunk in generate()]


# RunPod handler 비동기 방식으로 처리
def runpod_handler(event):
    logger.info(f"Received event: {event}")

    async def async_handler():
        endpoint = event["input"].get("endpoint", "")
        request_data = event["input"].get("params", {})
        headers = event["input"].get("headers", {})
        session_id = headers.get("x-session-id")

        # 세션별 인스턴스 관리 (비동기 방식)
        youtube_service, whisper_service, langchain_service = (
            await get_service_instances(session_id)
        )
        try:
            if endpoint == "rag_stream_chat":
                prompt = request_data.get("prompt")
                url_id = request_data.get("url_id")
                response = await rag_stream_chat(
                    prompt=prompt,
                    session_id=session_id,
                    url_id=url_id,
                    langchain_service=langchain_service,
                )
                return response

            elif endpoint == "get_title_hash":
                url = request_data.get("url")
                url_id = request_data.get("url_id")
                return await get_title_hash(
                    url=url, url_id=url_id, youtube_service=youtube_service
                )

            elif endpoint == "get_script_summary":
                url = request_data.get("url")
                url_id = request_data.get("url_id")
                return await get_script_summary(
                    url=url,
                    session_id=session_id,
                    url_id=url_id,
                    youtube_service=youtube_service,
                    whisper_service=whisper_service,
                    langchain_service=langchain_service,
                )
            else:
                return {"error": "Invalid endpoint"}

        except Exception as e:
            logger.error(f"Error during request processing: {e}")
            return {"error": str(e)}

    # 비동기 처리
    if asyncio.get_event_loop().is_running():
        return asyncio.create_task(async_handler())
    else:
        return asyncio.run(async_handler())


if __name__ == "__main__":
    runpod.serverless.start(
        {
            "handler": runpod_handler,
            "return_aggregate_stream": True,
        }
    )
