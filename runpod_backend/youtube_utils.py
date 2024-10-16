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
        video_id = self._extract_video_id(url)
        po_token = self._get_po_token(video_id)

        if po_token:
            print(f"po_token 생성 완료: {po_token}")
            return YouTube(url, use_po_token=True, po_token=po_token)
        else:
            print("po_token 생성 실패, 일반 모드로 시도합니다.")
            return YouTube(url)

    def _get_po_token(self, video_id: str):
        try:
            # youtube-po-token-generator 폴더로 경로 지정 후 실행
            command = [
                "node",
                "youtube-po-token-generator/index.js",
                "--video-id",
                video_id,
            ]
            result = subprocess.run(command, capture_output=True, text=True)

            if result.returncode == 0:
                return result.stdout.strip()
            else:
                print("po_token 생성 실패:", result.stderr)
                return None
        except Exception as e:
            print("po_token 생성 중 오류 발생:", str(e))
            return None

    def _extract_video_id(self, url: str):
        # 유튜브 URL에서 video_id 추출
        match = re.search(r"v=([^&]+)", url)
        return match.group(1) if match else None
