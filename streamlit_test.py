import streamlit as st
from datetime import datetime, timedelta, timezone
import uuid

# 한국 표준시(KST) 시간대 설정
kst = timezone(timedelta(hours=9))

# Streamlit 웹 애플리케이션 설정
st.set_page_config(layout="wide")
st.title("유튜브 요약 및 AI 채팅 (테스트 버전)")

# 초기 상태 설정
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# 추천 질문 예시
recommended_queries = [
    "이 영상의 핵심 포인트는 무엇인가요?",
    "주요 인물에 대한 설명을 해주세요.",
    "영상의 결론은 무엇인가요?",
    "이 영상과 관련된 주제는 무엇인가요?",
]

# 추천 질문 목록을 채팅 상단에 표시
st.subheader("AI 채팅")
with st.chat_message("assistant"):
    st.write("추천 질문 목록:")
    for query in recommended_queries:
        if st.button(query, key=query):
            # 사용자가 선택한 질문을 메시지로 추가
            st.session_state.messages.append({"role": "user", "content": query})

            # 더미 응답 생성
            bot_message = f"'{query}'에 대한 답변입니다. AI 응답이 여기에 표시됩니다."
            st.session_state.messages.append(
                {"role": "assistant", "content": bot_message}
            )

# 채팅 인터페이스
with st.container(height=800):
    messages_container = st.container(height=400)

    # 기존 채팅 기록 표시
    with messages_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
