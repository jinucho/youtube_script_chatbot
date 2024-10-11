from faster_whisper import WhisperModel
from config import Settings


class WhisperTranscriptionService:
    def __init__(self, settings: Settings):
        self.model = WhisperModel(
            "large-v3", device=settings.DEVICE, compute_type=settings.COMPUTE_TYPE
        )

    async def transcribe(self, audio_url: str):
        segments, info = self.model.transcribe(audio_url)
        transcript = self._process_segments(segments)
        return {"script": transcript, "language": info.language}

    def _process_segments(self, segments):
        transcript = []
        for segment in segments:
            transcript.append(
                {"start": segment.start, "end": segment.end, "text": segment.text}
            )
        return transcript
