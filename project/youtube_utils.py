from pytubefix import YouTube
import re
from config import settings


class YouTubeService:
    def __init__(self):
        self.settings = settings

    async def get_title_and_hashtags(self, url: str):
        yt = await self._create_youtube_instance(url)
        title = yt.title
        description = yt.description
        hashtags = re.findall(r"#\w+", description)
        return {"title": title, "hashtags": " ".join(hashtags)}

    async def get_video_info(self, url: str):
        yt = await self._create_youtube_instance(url)
        audio_stream = yt.streams.filter(only_audio=True).first()
        return {
            "title": yt.title,
            "audio_url": audio_stream.url if audio_stream else None,
        }

    async def _create_youtube_instance(self, url: str):
        return YouTube(url)
