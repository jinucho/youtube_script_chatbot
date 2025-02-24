import os
import math
import tempfile
import ffmpeg
import yt_dlp
from typing import Any, Dict, List

import soundfile as sf
from config import backup_data, settings
from faster_whisper import WhisperModel, BatchedInferencePipeline
from konlpy.tag import Okt
import concurrent.futures
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class WhisperTranscriptionService:
    def __init__(self, url_id):
        self.language = None
        self.okt = Okt()
        self.url_id = url_id
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

    async def transcribe(self, audio_url: str, prompt: dict = None, url_id: str = None) -> Dict[str, Any]:
        try:
            tagged = self.okt.pos(prompt.get("title", ""))
            filtered_words = [word for word, tag in tagged if tag in ("Noun", "Hashtag")]
        except:
            filtered_words = None
        
        script_info = backup_data.get(url_id=url_id).get("script_info", "")
        if script_info:
            return {
                "script": script_info.get("script"),
                "language": script_info.get("language"),
            }
        
        segments = self.process_with_progress(
            audio_url, prompt, filtered_words, chunk_duration=30, num_download_chunks=10
        )

        print("텍스트 추출 완료")
        backup_data.add_data(
            url_id=url_id,
            type="script_info",
            data={"script": segments, "language": self.language},
        )
        return {"script": segments, "language": self.language}

    def download_audio_parallel(self, url: str, temp_dir: str, num_chunks: int = 10) -> str:
        session = self.create_session()
        
        try:
            response = session.head(url, allow_redirects=True)
            total_size = int(response.headers.get("content-length", 0))
            
            if total_size == 0:
                response = session.get(url, stream=True)
                total_size = int(response.headers.get("content-length", 0))
                
            if total_size == 0:
                raise Exception("Failed to determine file size")
            
            chunk_size = total_size // num_chunks
            chunks = [(i * chunk_size, min((i + 1) * chunk_size - 1, total_size - 1)) for i in range(num_chunks)]
            
            download_args = [(url, start, end, i, temp_dir) for i, (start, end) in enumerate(chunks)]
            chunk_paths = []
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_chunks) as executor:
                futures = executor.map(self.download_chunk, download_args)
                chunk_paths = [(path, num) for path, num in futures if path is not None]
            
            if not chunk_paths:
                raise Exception("No chunks were downloaded successfully")
            
            chunk_paths.sort(key=lambda x: x[1])
            output_audio_path = os.path.join(temp_dir, "complete_audio.webm")
            
            with open(output_audio_path, "wb") as outfile:
                for chunk_path, _ in chunk_paths:
                    with open(chunk_path, "rb") as infile:
                        outfile.write(infile.read())
                    os.remove(chunk_path)
            
            output_wav_path = os.path.join(temp_dir, "complete_audio.wav")
            self.convert_to_wav(output_audio_path, output_wav_path)
            os.remove(output_audio_path)

            return output_wav_path
        
        except Exception as e:
            raise Exception(f"Error in parallel download: {str(e)}")

    def convert_to_wav(self, input_path: str, output_path: str) -> bool:
        try:
            stream = ffmpeg.input(input_path)
            stream = ffmpeg.output(stream, output_path, acodec="pcm_s16le", ar="16000", ac="1")
            ffmpeg.run(stream, capture_stdout=True, capture_stderr=True)
            return True
        except ffmpeg.Error as e:
            print("FFmpeg error:", e.stderr.decode())
            return False

    def process_with_progress(self, url: str, prompt: dict, filtered_words: str, chunk_duration: int = 30, num_download_chunks: int = 10) -> List[Dict[str, Any]]:
        with tempfile.TemporaryDirectory() as temp_dir:
            wav_path = self.download_audio_parallel(url, temp_dir, num_download_chunks)
            print("Download complete!")
            
            wav_info = sf.info(wav_path)
            total_duration = wav_info.duration
            total_chunks = math.ceil(total_duration / chunk_duration)
            
            all_segments = []
            for i in range(total_chunks):
                chunk_wav_path = os.path.join(temp_dir, f"chunk_{i}.wav")
                self.split_audio(wav_path, chunk_wav_path, i * chunk_duration, min(chunk_duration, total_duration - i * chunk_duration))
                
                model = WhisperModel("large-v3", device=settings.DEVICE, compute_type=settings.COMPUTE_TYPE)
                pipeline = BatchedInferencePipeline(model=model)
                segments, _ = pipeline.transcribe(chunk_wav_path)
                os.remove(chunk_wav_path)
                
                # 🔹 JSON 직렬화 가능하도록 변환
                formatted_segments = [
                    {"start": segment.start, "end": segment.end, "text": segment.text}
                    for segment in segments
                ]
                
                all_segments.extend(formatted_segments)
            
        return all_segments


    def split_audio(self, input_path: str, output_path: str, start_time: int, duration: int) -> str:
        stream = ffmpeg.input(input_path, ss=start_time, t=duration)
        stream = ffmpeg.output(stream, output_path, acodec="pcm_s16le", ar="16000", ac="1")
        ffmpeg.run(stream, quiet=True)
        return output_path
