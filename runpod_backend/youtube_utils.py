import yt_dlp
import re
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor
from config import backup_data, settings

DATA_PATH = settings.DATA_PATH


class YouTubeService:
    def __init__(self):
        self.executor = ThreadPoolExecutor()  # 비동기 실행을 위한 ThreadPoolExecutor

    async def get_video_data(self, url: str, url_id: str):
        # 해당 url_id로 저장된 데이터가 있는지 확인
        if os.path.isdir(os.path.join(DATA_PATH, f"{url_id}")):
            saved_data = backup_data.get(url_id)
            title = saved_data.get("title", "")
            hashtags = saved_data.get("hashtags", "")
            audio_url = saved_data.get("audio_url", "")
            return {
                "title_hashtags": {"title": title, "hashtags": hashtags},
                "audio_url": audio_url,
            }

        # YouTube 데이터 추출 (비동기 실행)
        yt_info = await self._fetch_video_info(url)

        # title과 hashtags 추출 및 저장
        title = yt_info.get("title", "")
        description = yt_info.get("description", "")
        hashtags = re.findall(r"#\w+", description)
        if len(hashtags) > 5:
            hashtags = hashtags[:5]
        hashtags = " ".join(hashtags)
        backup_data.add_title_and_hashtags(url_id, title, hashtags)

        # audio_url 추출 및 저장
        audio_url = yt_info.get("url") if yt_info else None
        backup_data.add_data(url_id, "audio_url", audio_url)

        # 최종 데이터 반환
        return {
            "title_hashtags": {"title": title, "hashtags": hashtags},
            "audio_url": audio_url,
        }

    async def _fetch_video_info(self, url: str):
        """ yt_dlp를 사용하여 유튜브 영상 정보 가져오기 (비동기 실행) """
        def get_info():
            ydl_opts = {"format": "bestaudio/best"}
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                return ydl.extract_info(url, download=False)

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self.executor, get_info)
