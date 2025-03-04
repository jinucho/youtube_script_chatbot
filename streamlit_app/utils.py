import os
import re
import smtplib
import time
from datetime import datetime, timedelta, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from io import BytesIO

import requests
from dotenv import load_dotenv

load_dotenv()
kst = timezone(timedelta(hours=9))

# RunPod 정보
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")
HEADERS = {
    "Authorization": f"Bearer {RUNPOD_API_KEY}",
    "Content-Type": "application/json",
}


def get_video_id(url):
    # 정규식을 통해 다양한 유튜브 링크에서 ID 추출
    match = re.search(
        r"(?:youtu\.be/|youtube\.com/(?:watch\?v=|embed/|v/|.+\?v=))([^&=%\?]{11})", url
    )
    return match.group(1) if match else None


def get_current_time():
    return datetime.now(kst).strftime("%H:%M")


def check_runpod_status(payload, RUNPOD_ENDPOINT_ID, interval=5):
    """
    RunPod 상태를 지속적으로 확인하여 'COMPLETED' 상태일 때 데이터를 반환.
    :param runpod_url: RunPod API 호출 URL
    :param headers: HTTP 요청 헤더
    :param payload: 요청에 필요한 데이터
    :param interval: 상태 확인 주기 (초)
    :return: 작업이 완료되면 결과 데이터 반환
    """
    RUNPOD_API_URL = f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}/runsync"
    response = requests.post(RUNPOD_API_URL, headers=HEADERS, json=payload)
    if response.status_code == 200:
        result = response.json()
        if result.get("status") in ["IN_PROGRESS", "IN_QUEUE"]:
            job_id = result.get("id")
            status_url = (
                f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}/status/{job_id}"
            )

            while True:
                status_response = requests.get(status_url, headers=HEADERS)
                if status_response.status_code == 200:
                    status_data = status_response.json()
                    if status_data.get("status") == "COMPLETED":
                        return status_data
                    else:
                        continue

                time.sleep(interval)  # 지정된 간격 후 다시 상태 확인
        elif result.get("status") == "COMPLETED":
            return result
        else:
            return response.json()


def send_feedback_email(feedback, session_id):
    sender_email = os.getenv("SENDER_EMAIL")
    sender_password = os.getenv("SENDER_PASSWORD")
    receiver_email = os.getenv("SENDER_EMAIL")

    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = f"새로운 사용자 피드백 (세션 ID: {session_id[:8]})"

    current_time = datetime.now(kst).strftime("%Y-%m-%d %H:%M:%S")
    body = f"피드백 시간: {current_time}\n"
    body += f"세션 ID: {session_id}\n\n"
    body += f"피드백 내용:\n{feedback}"

    message.attach(MIMEText(body, "plain"))

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(message)
        return True
    except Exception as e:
        print(f"이메일 전송 중 오류 발생: {e}")
        return False


def create_downloadable_file(session_state):
    # 텍스트 파일로 저장할 내용 구성
    title = f"제목: {session_state.title}"
    hashtags = f"해시태그: {session_state.hashtags}"
    joined_summary = "\n".join(session_state.summary)
    summary = f"\n[요약]\n{joined_summary}"

    # 스크립트를 텍스트로 변환
    transcript = "[스크립트]\n" + "\n".join(
        [
            f"{item['start']}초 - {item['end']}초: {item['text']}"
            for item in session_state.transcript
        ]
    )

    # 채팅 내역을 텍스트로 변환
    chat_history = "[채팅 내역]\n" + "\n".join(
        [
            f"{message['role']}: {message['content']}"
            for message in session_state.messages
        ]
    )

    # 모든 내용을 하나의 문자열로 결합
    data = f"{title}\n{hashtags}\n\n{summary}\n\n{transcript}\n\n{chat_history}"

    # 텍스트 파일을 바이트 형식으로 변환
    file_buffer = BytesIO()
    file_buffer.write(data.encode("utf-8"))
    file_buffer.seek(0)  # 파일 시작 위치로 포인터 이동
    return file_buffer
