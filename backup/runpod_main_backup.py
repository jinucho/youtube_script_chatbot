import asyncio
import logging
import multiprocessing
import os

import runpod
from langchain_utils import LangChainService
from whisper_transcription import WhisperTranscriptionService
from youtube_utils import YouTubeService
import warnings
import json

# 로그 설정
logging.basicConfig(level=logging.info)
logger = logging.getLogger(__name__)

warnings.filterwarnings(action="ignore")

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"  # OpenMP 스레드 수 제한

# 멀티프로세싱 설정 조정
multiprocessing.set_start_method("spawn", force=True)

youtube_service_instance = None
whisper_service_instance = None
langchain_service_cache = {}


def get_service_instances(session_id=None):
    global youtube_service_instance, whisper_service_instance

    # YouTubeService 인스턴스 생성 (싱글톤)
    if youtube_service_instance is None:
        youtube_service_instance = YouTubeService()

    # WhisperTranscriptionService 인스턴스 생성 (싱글톤)
    if whisper_service_instance is None:
        whisper_service_instance = WhisperTranscriptionService()

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


def concurrency_modifier(current_concurrency: int) -> int:
    # 최대 동시 실행 수가 5일 경우, 현재 동시 실행 수에 따라 조정
    return max(0, 5 - current_concurrency)


# 비동기 함수 정의
async def get_title_hash(url: str, youtube_service: YouTubeService):
    if not url or not isinstance(url, str):
        raise ValueError("Invalid URL format or missing URL")

    print(f"Received URL in get_title_hash: {url}")
    try:
        title_and_hashtags = await youtube_service.get_title_and_hashtags(url)
        print(f"Title and Hashtags for URL {url}: {title_and_hashtags}")
        return title_and_hashtags
    except Exception as e:
        logger.error(f"Error fetching title and hashtags for URL {url}: {e}")
        raise ValueError(f"Failed to fetch title and hashtags: {e}")


async def get_script_summary(
    url: str,
    session_id: str,
    youtube_service: YouTubeService,
    whisper_service: WhisperTranscriptionService,
    langchain_service: LangChainService,
):
    if not url or not isinstance(url, str):
        raise ValueError("Invalid URL format or missing URL")

    print(f"Received URL in get_script_summary: {url}")
    print(f"Session ID in get_script_summary: {session_id}")
    try:
        video_info = await youtube_service.get_video_info(url)
        print(f"[Session {session_id}] Video info for URL {url}: {video_info}")

        transcript = await whisper_service.transcribe(video_info["audio_url"])
        print(
            f"[Session {session_id}] Transcript for video {video_info['audio_url'][:10]}: {transcript.get('script')[:3]}"
        )

        summary = await langchain_service.summarize(transcript)
        print(f"[Session {session_id}] Summary for transcript: {summary[:10]}")

        return {
            "summary_result": summary,
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
    prompt: str, session_id: str, langchain_service: LangChainService
):
    if not session_id:
        raise ValueError("Session ID is required")

    if not prompt:
        raise ValueError("Prompt is required")

    async def generate():
        try:
            async for chunk in langchain_service.stream_chat(prompt):
                yield {"content": chunk}
            yield {"content": "[DONE]"}
        except Exception as e:
            logger.error(f"[Session {session_id}] Error during stream_chat: {e}")
            yield {"error": str(e)}
            yield {"content": "[DONE]"}

    return [chunk async for chunk in generate()]


# RunPod handler
def runpod_handler(event):
    print(f"Received event: {event}")

    async def async_handler():
        endpoint = event["input"].get("endpoint", "/")
        method = event["input"].get("method", "POST").upper()
        request_data = event["input"].get("params", {})
        headers = event["input"].get("headers", {})
        session_id = headers.get("x-session-id")

        # session_id 기준으로 서비스 인스턴스 가져오기
        youtube_service, whisper_service, langchain_service = get_service_instances(
            session_id
        )

        try:
            if method == "POST":
                if endpoint == "/rag_stream_chat":
                    prompt = request_data.get("prompt")
                    response = await rag_stream_chat(
                        prompt=prompt,
                        session_id=session_id,
                        langchain_service=langchain_service,
                    )
                    return response

            elif method == "GET":
                if endpoint == "/get_title_hash":
                    url = request_data.get("url")
                    return await get_title_hash(url, youtube_service=youtube_service)

                elif endpoint == "/get_script_summary":
                    url = request_data.get("url")
                    return await get_script_summary(
                        url=url,
                        session_id=session_id,
                        youtube_service=youtube_service,
                        whisper_service=whisper_service,
                        langchain_service=langchain_service,
                    )
                else:
                    return {"error": "Invalid endpoint or method"}

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
            "concurrency_modifier": concurrency_modifier,
            "return_aggregate_stream": True,
        }
    )
