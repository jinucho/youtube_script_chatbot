import os
import smtplib
from datetime import datetime, timedelta, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
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
