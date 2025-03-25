import yt_dlp
import re
import os
import asyncio
import tempfile
import json
import base64
from concurrent.futures import ThreadPoolExecutor
from config import backup_data, settings
import logging

DATA_PATH = settings.DATA_PATH


class YouTubeService:
    def __init__(self):
        self.executor = ThreadPoolExecutor()  # 비동기 실행을 위한 ThreadPoolExecutor
        # 환경 변수에서 인코딩된 쿠키 가져오기
        self.youtube_cookies = os.environ.get("YOUTUBE_COOKIES", "")

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
        yt_info = self._fetch_video_info(url)

        # title과 hashtags 추출 및 저장
        title = yt_info.get("title", "")
        description = yt_info.get("description", "")
        hashtags = re.findall(r"#\w+", description)
        if len(hashtags) > 5:
            hashtags = hashtags[:5]
        hashtags = " ".join(hashtags)
        backup_data.add_title_and_hashtags(url_id, title, hashtags)

        # audio_url 추출 및 저장
        audio_url = None
        for i in yt_info.get("requested_formats"):
            if i.get("vcodec") == "none":
                audio_url = i.get("url")
                break
        backup_data.add_data(url_id, "audio_url", audio_url)

        # 최종 데이터 반환
        return {
            "title_hashtags": {"title": title, "hashtags": hashtags},
            "audio_url": audio_url,
        }

    def _fetch_video_info(self, url: str):
        """yt_dlp를 사용하여 유튜브 영상 정보 가져오기 (비동기 실행)"""

        def get_info():
            try:
                if self.youtube_cookies:
                    # base64로 인코딩된 쿠키 문자열을 디코딩
                    cookie_data = base64.b64decode(self.youtube_cookies).decode("utf-8")

                    # 쿠키가 JSON 형식인지 확인하고 Netscape 형식으로 변환
                    json_cookies = json.loads(cookie_data)

                    # Netscape 형식으로 변환
                    netscape_cookies = "# Netscape HTTP Cookie File\n"
                    for cookie in json_cookies:
                        if (
                            "domain" in cookie
                            and "path" in cookie
                            and "name" in cookie
                            and "value" in cookie
                        ):
                            secure = "TRUE" if cookie.get("secure", False) else "FALSE"
                            http_only = (
                                "TRUE" if cookie.get("httpOnly", False) else "FALSE"
                            )
                            expires = str(int(cookie.get("expirationDate", 0)))
                            netscape_cookies += f"{cookie['domain']}\tTRUE\t{cookie['path']}\t{secure}\t{expires}\t{cookie['name']}\t{cookie['value']}\n"

                    cookie_data = netscape_cookies

                ydl_opts = {
                    "quiet": True,
                    "no_warnings": True,
                    "extract_flat": True,  # 기본 정보만 추출하도록 변경
                    "nocheckcertificate": True,
                    "ignoreerrors": True,
                    "no_color": True,
                    "socket_timeout": 30,  # 소켓 타임아웃 설정
                    "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                }

                # 쿠키 파일이 있으면 옵션에 추가
                if cookie_data:
                    ydl_opts["cookiefile"] = cookie_data
                ydl = yt_dlp.YoutubeDL(ydl_opts)
                return ydl.extract_info(url, download=False)
            except Exception as e:
                logging.error(f"Error fetching video info: {e}")
                return None

        return get_info()
