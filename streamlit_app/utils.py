import os
import smtplib
from datetime import datetime, timedelta, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from io import BytesIO

from dotenv import load_dotenv

load_dotenv()
kst = timezone(timedelta(hours=9))


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
    title = f"제목: {session_state.title}\n"
    hashtags = f"해시태그: {session_state.hashtags}\n"

    summary = f"\n[요약]\n{session_state.summary}\n"

    # 스크립트를 텍스트로 변환
    transcript = (
        "[스크립트]\n"
        + "\n".join(
            [
                f"{item['start']}초 - {item['end']}초: {item['text']}"
                for item in session_state.transcript
            ]
        )
        + "\n"
    )

    # 채팅 내역을 텍스트로 변환
    chat_history = "[채팅 내역]\n" + "\n".join(
        [
            f"{message['role']}: {message['content']}"
            for message in session_state.messages
        ]
    )

    # 모든 내용을 하나의 문자열로 결합
    data = title + hashtags + summary + transcript + chat_history

    # 텍스트 파일을 바이트 형식으로 변환
    file_buffer = BytesIO()
    file_buffer.write(data.encode("utf-8"))
    file_buffer.seek(0)  # 파일 시작 위치로 포인터 이동
    return file_buffer
