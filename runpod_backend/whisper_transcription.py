import concurrent.futures
import math
import os
import tempfile
from typing import Any, Dict, List

import ffmpeg
import requests
import soundfile as sf
from config import settings
from faster_whisper import BatchedInferencePipeline, WhisperModel
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class WhisperTranscriptionService:
    def __init__(self):
        model = WhisperModel(
            "large-v3", device=settings.DEVICE, compute_type=settings.COMPUTE_TYPE
        )
        self.model = BatchedInferencePipeline(model=model)
        self.language = None
        print("Whisper 모델 초기화 완료")

    def create_session(self):
        session = requests.Session()
        retry = Retry(
            total=5, backoff_factor=0.1, status_forcelist=[500, 502, 503, 504]
        )
        adapter = HTTPAdapter(max_retries=retry, pool_connections=100, pool_maxsize=100)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

    def download_chunk(self, args):
        url, start, end, chunk_number, temp_dir = args

        headers = {"Range": f"bytes={start}-{end}"}
        session = self.create_session()

        try:
            response = session.get(url, headers=headers, stream=True)
            chunk_path = os.path.join(temp_dir, f"chunk_{chunk_number:04d}")

            with open(chunk_path, "wb") as f:
                for data in response.iter_content(chunk_size=8192):
                    f.write(data)

            return chunk_path, chunk_number
        except Exception as e:
            print(f"Error downloading chunk {chunk_number}: {str(e)}")
            return None, chunk_number

    def _single_stream_download(self, url: str, temp_dir: str) -> str:
        """단일 스트림으로 파일을 다운로드합니다."""
        session = self.create_session()
        output_path = os.path.join(temp_dir, "complete_audio.mp4")

        try:
            with session.get(url, stream=True) as response:
                response.raise_for_status()
                with open(output_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
            return output_path
        except Exception as e:
            raise Exception(f"Failed to download file: {str(e)}")

    def parallel_download(self, url: str, temp_dir: str, num_chunks: int = 10) -> str:
        """병렬 다운로드를 시도하고, 실패 시 단일 스트림으로 폴백"""
        session = self.create_session()

        try:
            # HEAD 요청으로 파일 크기 확인 시도
            response = session.head(url, allow_redirects=True)
            total_size = int(response.headers.get("content-length", 0))

            # HEAD 요청이 실패하면 GET 요청으로 시도
            if total_size == 0:
                response = session.get(url, stream=True)
                total_size = int(response.headers.get("content-length", 0))

            # 파일 크기를 여전히 확인할 수 없는 경우 단일 스트림으로 다운로드
            if total_size == 0:
                print(
                    "Warning: Could not determine file size. Falling back to single stream download."
                )
                return self._single_stream_download(url, temp_dir)
            print("Starting parallel download...")
            chunk_size = total_size // num_chunks
            chunks = []

            for i in range(num_chunks):
                start = i * chunk_size
                end = start + chunk_size - 1 if i < num_chunks - 1 else total_size - 1
                chunks.append((start, end))

            download_args = [
                (url, start, end, i, temp_dir) for i, (start, end) in enumerate(chunks)
            ]

            chunk_paths = []
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=num_chunks
            ) as executor:
                futures = executor.map(self.download_chunk, download_args)
                chunk_paths = [(path, num) for path, num in futures if path is not None]

            if not chunk_paths:
                raise Exception("No chunks were downloaded successfully")

            chunk_paths.sort(key=lambda x: x[1])
            output_path = os.path.join(temp_dir, "complete_audio.mp4")

            with open(output_path, "wb") as outfile:
                for chunk_path, _ in chunk_paths:
                    with open(chunk_path, "rb") as infile:
                        outfile.write(infile.read())
                    os.remove(chunk_path)

            return output_path

        except Exception as e:
            print(
                f"Error in parallel download: {str(e)}. Falling back to single stream download."
            )
            return self._single_stream_download(url, temp_dir)

    def convert_to_wav(self, input_path: str, output_path: str) -> bool:
        try:
            stream = ffmpeg.input(input_path)
            stream = ffmpeg.output(
                stream, output_path, acodec="pcm_s16le", ar="16000", ac="1"
            )
            ffmpeg.run(stream, capture_stdout=True, capture_stderr=True)
            return True
        except ffmpeg.Error as e:
            print("FFmpeg error:", e.stderr.decode())
            return False

    def process_audio_chunk(self, chunk_data: tuple) -> List[Dict[str, Any]]:
        audio_path, start_time, duration = chunk_data
        try:
            segments, info = self.model.transcribe(
                audio_path,
                beam_size=15,
                batch_size=32,
                temperature=0.5,
                word_timestamps=True,
                initial_prompt="This audio may contain technical terms and English words; if present, retain English terms as is.",  # 영어와 기술 용어가 있을 경우 그대로 유지 요청
                temperature=0.3,
                repetition_penalty=2,
            )
            if info and hasattr(info, "language"):
                self.language = info.language
            return self._process_segments(segments, start_time)
        except Exception as e:
            print(f"Error processing chunk at {start_time}: {str(e)}")
            return []

    def _process_segments(
        self, segments, start_time: float = 0
    ) -> List[Dict[str, Any]]:
        transcript = []
        for segment in segments:
            transcript.append(
                {
                    "start": round(segment.start + start_time, 2),
                    "end": round(segment.end + start_time, 2),
                    "text": segment.text,
                }
            )
        return transcript

    async def process_with_progress(
        self, url: str, chunk_duration: int = 30, num_download_chunks: int = 10
    ) -> List[Dict[str, Any]]:
        with tempfile.TemporaryDirectory() as temp_dir:
            print("Starting download...")
            mp4_path = self.parallel_download(url, temp_dir, num_download_chunks)
            print("Download complete!")

            wav_path = os.path.join(temp_dir, "audio.wav")
            if not self.convert_to_wav(mp4_path, wav_path):
                raise Exception("Failed to convert audio to WAV format")

            wav_info = sf.info(wav_path)
            total_duration = wav_info.duration
            total_chunks = math.ceil(total_duration / chunk_duration)

            chunks_data = []
            for i in range(total_chunks):
                start_time = i * chunk_duration
                chunk_wav_path = os.path.join(temp_dir, f"chunk_{i}.wav")

                duration = min(chunk_duration, total_duration - start_time)
                stream = ffmpeg.input(wav_path, ss=start_time, t=duration)
                stream = ffmpeg.output(
                    stream, chunk_wav_path, acodec="pcm_s16le", ar="16000", ac="1"
                )
                ffmpeg.run(stream, quiet=True)

                chunks_data.append((chunk_wav_path, start_time, duration))

            all_segments = []
            for chunk_data in chunks_data:
                segments = self.process_audio_chunk(chunk_data)
                all_segments.extend(segments)

                if os.path.exists(chunk_data[0]):
                    os.remove(chunk_data[0])

        return all_segments

    async def transcribe(self, audio_url: str) -> Dict[str, Any]:
        try:
            segments = await self.process_with_progress(
                audio_url, chunk_duration=30, num_download_chunks=10
            )

            print("텍스트 추출 완료")

            return {"script": segments, "language": self.language}
        except Exception as e:
            print(f"Error in transcribe: {str(e)}")
            raise
