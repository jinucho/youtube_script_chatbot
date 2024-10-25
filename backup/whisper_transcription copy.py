from faster_whisper import WhisperModel, BatchedInferencePipeline
from config import settings
from tqdm import tqdm
import soundfile as sf
import math
import requests
import tempfile
import os
import concurrent.futures
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
import threading
import ffmpeg
from typing import Dict, List, Any


class ProgressBar:
    def __init__(self, total_size, desc="Downloading"):
        self.pbar = tqdm(total=total_size, unit="iB", unit_scale=True, desc=desc)
        self.lock = threading.Lock()

    def update(self, size):
        with self.lock:
            self.pbar.update(size)

    def close(self):
        self.pbar.close()


class WhisperTranscriptionService:
    def __init__(self):
        model = WhisperModel(
            "large-v3", device=settings.DEVICE, compute_type=settings.COMPUTE_TYPE
        )
        self.model = BatchedInferencePipeline(model=model)
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
        url, start, end, chunk_number, temp_dir, progress_bar = args

        headers = {"Range": f"bytes={start}-{end}"}
        session = self.create_session()

        try:
            response = session.get(url, headers=headers, stream=True)
            chunk_path = os.path.join(temp_dir, f"chunk_{chunk_number:04d}")

            with open(chunk_path, "wb") as f:
                for data in response.iter_content(chunk_size=8192):
                    size = f.write(data)
                    progress_bar.update(size)

            return chunk_path, chunk_number
        except Exception as e:
            print(f"Error downloading chunk {chunk_number}: {str(e)}")
            return None, chunk_number

    def parallel_download(self, url: str, temp_dir: str, num_chunks: int = 10) -> str:
        session = self.create_session()
        response = session.head(url)
        total_size = int(response.headers.get("content-length", 0))

        if total_size == 0:
            raise ValueError("Could not determine file size")

        chunk_size = total_size // num_chunks
        chunks = []

        for i in range(num_chunks):
            start = i * chunk_size
            end = start + chunk_size - 1 if i < num_chunks - 1 else total_size - 1
            chunks.append((start, end))

        progress_bar = ProgressBar(total_size, "Parallel downloading")

        download_args = [
            (url, start, end, i, temp_dir, progress_bar)
            for i, (start, end) in enumerate(chunks)
        ]

        chunk_paths = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_chunks) as executor:
            futures = executor.map(self.download_chunk, download_args)
            chunk_paths = [(path, num) for path, num in futures if path is not None]

        progress_bar.close()

        chunk_paths.sort(key=lambda x: x[1])
        output_path = os.path.join(temp_dir, "complete_audio.mp4")

        with open(output_path, "wb") as outfile:
            for chunk_path, _ in chunk_paths:
                with open(chunk_path, "rb") as infile:
                    outfile.write(infile.read())
                os.remove(chunk_path)

        return output_path

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
                beam_size=5,
                batch_size=32,
                word_timestamps=True,
                initial_prompt=None,
            )
            self.language = info.language
            return self._process_segments(segments, start_time)
        except Exception as e:
            print(f"Error processing chunk at {start_time}: {str(e)}")
            return []

    def _process_segments(
        self, segments, start_time: float = 0
    ) -> List[Dict[str, Any]]:
        transcript = []
        for segment in tqdm(segments, desc="Processing segments"):
            transcript.append(
                {
                    "start": round(segment.start + start_time, 2),
                    "end": round(segment.end + start_time, 2),
                    "text": segment.text,
                    "words": (
                        [
                            {
                                "start": round(word.start + start_time, 2),
                                "end": round(word.end + start_time, 2),
                                "word": word.word,
                                "probability": round(word.probability, 4),
                            }
                            for word in segment.words
                        ]
                        if hasattr(segment, "words")
                        else []
                    ),
                }
            )
        return transcript

    async def process_with_progress(
        self, url: str, chunk_duration: int = 30, num_download_chunks: int = 10
    ) -> List[Dict[str, Any]]:
        with tempfile.TemporaryDirectory() as temp_dir:
            print("Starting parallel download...")
            mp4_path = self.parallel_download(url, temp_dir, num_download_chunks)
            print("Download complete!")

            wav_path = os.path.join(temp_dir, "audio.wav")
            if not self.convert_to_wav(mp4_path, wav_path):
                raise Exception("Failed to convert audio to WAV format")

            wav_info = sf.info(wav_path)
            total_duration = wav_info.duration

            total_chunks = math.ceil(total_duration / chunk_duration)
            pbar = tqdm(total=total_chunks, desc="Processing audio chunks")

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
                pbar.update(1)

                if os.path.exists(chunk_data[0]):
                    os.remove(chunk_data[0])

            pbar.close()

        return all_segments

    async def transcribe(self, audio_url: str) -> Dict[str, Any]:
        segments = await self.process_with_progress(
            audio_url, chunk_duration=30, num_download_chunks=10
        )

        language = self.language  # 또는 실제 감지된 언어
        print("텍스트 추출 완료")

        return {"script": segments, "language": language}
