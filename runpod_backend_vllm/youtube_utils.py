from pytubefix import YouTube
import re
import os
from config import backup_data, settings


DATA_PATH = settings.DATA_PATH


class YouTubeService:
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

        # YouTube 인스턴스 생성 및 데이터 추출
        yt = await self._create_youtube_instance(url)
        print("영상 정보 확인")

        # title과 hashtags 추출 및 저장
        title = yt.title
        description = yt.description
        hashtags = re.findall(r"#\w+", description)
        if len(hashtags) > 5:
            hashtags = hashtags[:5]
        hashtags = " ".join(re.findall(r"#\w+", description))
        backup_data.add_title_and_hashtags(url_id, title, hashtags)

        # audio_url 추출 및 저장
        audio_stream = yt.streams.filter(only_audio=True).first()
        audio_url = audio_stream.url if audio_stream else None
        backup_data.add_data(url_id, "audio_url", audio_url)

        # 최종 데이터 반환
        return {
            "title_hashtags": {"title": title, "hashtags": hashtags},
            "audio_url": audio_url,
        }

    async def _create_youtube_instance(self, url: str):
        print("YouTube 인스턴스 생성 완료")
        return YouTube(url)
