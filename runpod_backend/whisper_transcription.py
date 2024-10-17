from faster_whisper import WhisperModel, BatchedInferencePipeline
from config import settings


class WhisperTranscriptionService:
    def __init__(self):
        model = WhisperModel(
            "large-v3", device=settings.DEVICE, compute_type=settings.COMPUTE_TYPE
        )
        self.model = BatchedInferencePipeline(model=model)

    async def transcribe(self, audio_url: str):
        segments, info = self.model.transcribe(audio_url, batch_size=16,repetition_penalty=1.5,log_progress=True,beam_size=15)
        transcript = self._process_segments(segments)
        print("텍스트 추출 완료")
        return {"script": transcript, "language": info.language}

    def _process_segments(self, segments):
        transcript = []
        for segment in segments:
            transcript.append(
                {
                    "start": round(segment.start, 2),
                    "end": round(segment.end, 2),
                    "text": segment.text,
                }
            )
        print("스크립트 정리 완료")
        return transcript
