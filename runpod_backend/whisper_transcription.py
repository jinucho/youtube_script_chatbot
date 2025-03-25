import logging
from typing import Any, Dict, List

from config import backup_data, settings
from faster_whisper import WhisperModel, BatchedInferencePipeline
from konlpy.tag import Okt

logger = logging.getLogger(__name__)


class WhisperTranscriptionService:
    def __init__(self, url_id):
        self.language = None
        self.okt = Okt()
        self.url_id = url_id
        self.model = WhisperModel(
            "large-v3",
            device=settings.DEVICE,
            compute_type=settings.COMPUTE_TYPE,
        )
        self.pipeline = BatchedInferencePipeline(model=self.model)
        print("Whisper 모델 초기화 완료")

    async def transcribe(
        self, audio_url: str, prompt: dict = None, url_id: str = None
    ) -> Dict[str, Any]:
        try:
            tagged = self.okt.pos(prompt.get("title", ""))
            filtered_words = [
                word for word, tag in tagged if tag in ("Noun", "Hashtag")
            ]
            filtered_words = " ".join(filtered_words)
        except:
            filtered_words = None

        script_info = backup_data.get(url_id=url_id).get("script_info", "")
        if script_info:
            return {
                "script": script_info.get("script"),
                "language": script_info.get("language"),
            }

        segments = self.process_with_progress(audio_url, filtered_words)

        print("텍스트 추출 완료")
        backup_data.add_data(
            url_id=url_id,
            type="script_info",
            data={"script": segments, "language": self.language},
        )
        return {"script": segments, "language": self.language}

    def process_with_progress(
        self,
        url: str,
        filtered_words: str,
    ) -> List[Dict[str, Any]]:

        segments, _ = self.pipeline.transcribe(url, initial_prompt=filtered_words)
        logger.info(f"음성 추출 완료")

        formatted_segments = [
            {"start": segment.start, "end": segment.end, "text": segment.text}
            for segment in segments
        ]
        logger.info(f"텍스트 추출 완료")
        return formatted_segments
