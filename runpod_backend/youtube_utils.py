# from pytubefix import YouTube
# import re


# class YouTubeService:
#     async def get_title_and_hashtags(self, url: str):
#         yt = await self._create_youtube_instance(url)
#         print("영상 정보 확인")
#         title = yt.title
#         description = yt.description
#         hashtags = re.findall(r"#\w+", description)
#         return {"title": title, "hashtags": " ".join(hashtags)}

#     async def get_video_info(self, url: str):
#         yt = await self._create_youtube_instance(url)
#         audio_stream = yt.streams.filter(only_audio=True).first()
#         print("음성 추출 완료")
#         return {
#             "title": yt.title,
#             "audio_url": audio_stream.url if audio_stream else None,
#         }

#     async def _create_youtube_instance(self, url: str):
#         print("YouTube 인스턴스 생성 완료")
#         return YouTube(url)

import os
from pytubefix import YouTube


class YouTubeService:
    async def get_audio_file(self, url: str, output_dir: str = "./downloads"):
        yt = await self._create_youtube_instance(url)
        audio_stream = yt.streams.filter(only_audio=True).first()

        if not audio_stream:
            print("오디오 스트림을 찾을 수 없습니다.")
            return None

        # 다운로드할 경로 설정
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 파일명을 YouTube 제목 기반으로 설정
        file_path = os.path.join(output_dir, f"{yt.title}.mp3")
        audio_stream.download(output_path=output_dir, filename=f"{yt.title}.mp3")

        print(f"오디오 파일 다운로드 완료: {file_path}")
        return file_path

    async def _create_youtube_instance(self, url: str):
        print("YouTube 인스턴스 생성 완료")
        return YouTube(url)
