import json
import os
import time
import uuid
from datetime import datetime, timedelta, timezone
import streamlit as st
import requests
from dotenv import load_dotenv

load_dotenv()

from mail import send_feedback_email

# RunPod 정보
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")
RUNPOD_ENDPOINT_ID = os.getenv("RUNPOD_ENDPOINT_ID")
RUNPOD_API_URL = f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}/runsync"
HEADERS = {
    "Authorization": f"Bearer {RUNPOD_API_KEY}",
    "Content-Type": "application/json",
}

# 한국 표준시(KST) 시간대 설정
kst = timezone(timedelta(hours=9))

# Streamlit 웹 애플리케이션 설정
st.set_page_config(layout="wide")  # 전체 레이아웃을 넓게 설정
st.title("유튜브 요약 및 AI 채팅")

st.write("참고사항 : 첫 시작 시 시간이 소요 됩니다.")
st.write("주의사항 : 1분동안 아무 요청이 없을 경우 세션이 종료 됩니다.")

# 초기 상태 설정
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_input" not in st.session_state:
    st.session_state.last_input = ""
if "title" not in st.session_state:
    st.session_state.title = ""
if "hashtags" not in st.session_state:
    st.session_state.hashtags = ""
if "video_id" not in st.session_state:
    st.session_state.video_id = ""
if "summary" not in st.session_state:
    st.session_state.summary = ""
if "transcript" not in st.session_state:
    st.session_state.transcript = []
if "session_id" not in st.session_state:
    # 세션 ID 생성 (각 사용자마다 고유한 세션 ID를 생성)
    st.session_state.session_id = str(uuid.uuid4())


def check_runpod_status(payload, interval=5):
    """
    RunPod 상태를 지속적으로 확인하여 'COMPLETED' 상태일 때 데이터를 반환.
    :param runpod_url: RunPod API 호출 URL
    :param headers: HTTP 요청 헤더
    :param payload: 요청에 필요한 데이터
    :param interval: 상태 확인 주기 (초)
    :return: 작업이 완료되면 결과 데이터 반환
    """
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


# 유튜브 URL 입력 받기
url = st.text_input("유튜브 URL을 입력하세요:", key="youtube_url")

# URL 입력 및 스크립트 추출을 위한 버튼 클릭 상태 확인
if st.button("스크립트 추출"):
    if url:
        if "youtu" not in url:
            st.warning("유효한 유튜브 URL을 입력하세요.")
        else:
            # get_title_hash 엔드포인트 호출
            payload = {
                "input": {
                    "endpoint": "/get_title_hash",
                    "method": "GET",
                    "params": {"url": url},
                }
            }
            data = check_runpod_status(payload)
            st.session_state.title = data.get("output", {}).get("title", "제목")
            st.session_state.hashtags = data.get("output", {}).get("hashtags", "")
            st.session_state.video_id = url.split("/")[-1]

            with st.spinner("요약 중입니다..."):
                # get_script_summary 엔드포인트 호출
                payload = {
                    "input": {
                        "endpoint": "/get_script_summary",
                        "method": "GET",
                        "headers": {"x-session-id": st.session_state.session_id},
                        "params": {"url": url},
                    }
                }

                # 상태를 직접 확인하여 작업 완료 시까지 대기
                summary_response = check_runpod_status(payload)

                if summary_response:
                    summary_data = summary_response.get("output", {})
                    st.session_state.summary = summary_data.get("summary_result", "")
                    st.session_state.language = summary_data.get("language", "")
                    st.session_state.transcript = summary_data.get("script", [])
                else:
                    st.error("스크립트 요약에 실패했습니다.")

# URL이 입력되었고, 데이터가 session_state에 저장된 경우 표시
if st.session_state.title:  # 타이틀이 존재하는 경우에만 레이아웃 표시
    col1, col2 = st.columns(2)

    with col1:
        st.subheader(f"제목 : {st.session_state.title}")
        st.write(st.session_state.hashtags)

        if st.session_state.video_id:
            st.markdown(
                f'<iframe width="100%" height="600" src="https://www.youtube.com/embed/{st.session_state.video_id}" '
                f'frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" '
                f"allowfullscreen></iframe>",
                unsafe_allow_html=True,
            )

        st.subheader("요약내용")
        st.write(st.session_state.summary)

        with st.expander("스크립트 보기", expanded=False):
            if st.session_state.transcript:
                with st.container(height=400):
                    for item in st.session_state.transcript:
                        st.write(f"{item['start']}초 - {item['end']}초: {item['text']}")

    with col2:
        st.subheader("AI 채팅")

        # 메시지를 표시할 고정 컨테이너
        messages_container = st.container(height=800)

        # 채팅 입력창을 위한 컨테이너 (가장 하단에 배치)
        input_container = st.container()

        # 채팅 입력 처리 (먼저 처리하되 UI는 하단에 표시)
        with input_container:
            prompt = st.chat_input("메시지를 입력하세요")

        # 메시지 표시 (채팅 이력)
        with messages_container:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.write(message["content"])

            # 새 메시지 처리
            if prompt:
                current_time = datetime.now(kst).strftime("%H:%M")

                # 사용자 메시지 추가 및 표시
                user_message = f"{prompt} ({current_time})"
                with st.chat_message("user"):
                    st.write(user_message)
                st.session_state.messages.append(
                    {"role": "user", "content": user_message}
                )

                # 봇 응답 생성
                with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    bot_message = ""

                    # RunPod API 호출
                    payload = {
                        "input": {
                            "endpoint": "/rag_stream_chat",
                            "method": "POST",
                            "headers": {"x-session-id": st.session_state.session_id},
                            "params": {"prompt": prompt},
                        }
                    }

                    try:
                        chunks = check_runpod_status(payload)
                        for chunk in chunks.get("output"):
                            if "content" in chunk:
                                content = chunk["content"]
                                if content == "[DONE]":
                                    break
                                bot_message += content
                                message_placeholder.write(f"{bot_message}▌")
                                time.sleep(0.05)

                            elif "error" in chunk:
                                st.error(f"Error: {chunk['error']}")
                                break

                        # 최종 봇 메시지 저장
                        final_message = f"{bot_message} ({current_time})"
                        message_placeholder.write(final_message)
                        st.session_state.messages.append(
                            {"role": "assistant", "content": final_message}
                        )

                    except requests.RequestException as e:
                        st.error(f"Request failed: {str(e)}")
                    except json.JSONDecodeError:
                        st.error("Failed to decode response")

st.markdown("---")
st.header("피드백을 보내주세요.")
feedback = st.text_area("사용 시 불편한 점이나, 오류가 있었다면 알려주세요.:")
if st.button("전송"):
    if feedback:
        if send_feedback_email(feedback, st.session_state.session_id):
            st.success("피드백 감사합니다!")
        else:
            st.error("전송 중 오류가 발생했습니다. 나중에 다시 시도해 주세요.")
    else:
        st.warning("피드백을 입력해 주세요.")
