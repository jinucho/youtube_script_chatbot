from faster_whisper import WhisperModel, BatchedInferencePipeline
from config import settings


class WhisperTranscriptionService:
    def __init__(self):
        model = WhisperModel(
            "large-v3", device=settings.DEVICE, compute_type=settings.COMPUTE_TYPE
        )
        self.model = BatchedInferencePipeline(model=model)  # 배치 모델일 경우
        print("Whisper 모델 초기화 완료")

    async def transcribe(self, audio_url: str):
        # 제너레이터에서 데이터를 리스트로 변환
        segments, info = self.model.transcribe(
            audio_url,
            batch_size=16,  # 배치 모델인 경우
            repetition_penalty=1.5,
            beam_size=10,
            patience=2,
            no_repeat_ngram_size=4,
        )

        # 제너레이터 처리하여 스크립트 생성
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
