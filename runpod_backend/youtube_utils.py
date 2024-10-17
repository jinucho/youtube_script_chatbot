from pytubefix import YouTube
import re
import subprocess


class YouTubeService:
    async def get_title_and_hashtags(self, url: str):
        yt = await self._create_youtube_instance(url)
        print("영상 정보 확인")
        title = yt.title
        description = yt.description
        hashtags = re.findall(r"#\w+", description)
        return {"title": title, "hashtags": " ".join(hashtags)}

    async def get_video_info(self, url: str):
        yt = await self._create_youtube_instance(url)
        audio_stream = yt.streams.filter(only_audio=True).first()
        print("음성 추출 완료")
        return {
            "title": yt.title,
            "audio_url": audio_stream.url if audio_stream else None,
        }

    async def _create_youtube_instance(self, url: str):
        return YouTube(url)
