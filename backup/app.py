import backup.app_streamlit as st
import requests
from datetime import datetime, timedelta, timezone
import uuid
import json

# 한국 표준시(KST) 시간대 설정
kst = timezone(timedelta(hours=9))

# Streamlit 웹 애플리케이션 설정
st.set_page_config(layout="wide")  # 전체 레이아웃을 넓게 설정
st.title("유튜브 요약 및 AI 채팅")

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

# CSS 스타일 정의
st.markdown(
    """
<style>
.chat-container {
    height: 800px;
    overflow-y: auto;
    border: 1px solid #ddd;
    padding: 10px;
    margin-bottom: 10px;
}
.user-message {
    background-color: #DCF8C6;
    color: black;
    border-radius: 10px;
    padding: 8px;
    margin: 5px;
    max-width: 70%;
    float: right;
    clear: both;
}
.bot-message {
    background-color: #17EAE4;
    color: black;
    border-radius: 10px;
    padding: 8px;
    margin: 5px;
    max-width: 70%;
    float: left;
    clear: both;
}
</style>
""",
    unsafe_allow_html=True,
)


# 채팅 메시지 표시 함수
def display_message(role, content):
    message_class = "user-message" if role == "user" else "bot-message"
    return f'<div class="{message_class}">{content}</div>'


# 채팅 처리 함수
def process_input():
    if (
        st.session_state.chat_input
        and st.session_state.chat_input != st.session_state.get("last_input", "")
    ):
        current_time = datetime.now(kst).strftime("%H:%M")
        user_message = f"{st.session_state.chat_input} ({current_time})"
        st.session_state.messages.append({"role": "user", "content": user_message})

        # 봇 응답 생성 및 추가 (스트리밍)
        with st.spinner("AI가 응답을 생성 중입니다..."):
            url = "http://0.0.0.0:8080/rag_stream_chat"
            headers = {
                "Content-Type": "application/json",
                "x-session-id": st.session_state.session_id,  # session_id를 헤더에 포함
            }
            data = {"prompt": st.session_state.chat_input}

            with requests.post(
                url, headers=headers, json=data, stream=True
            ) as response:
                bot_message = ""
                for chunk in response.iter_lines(decode_unicode=True):
                    if chunk:
                        chunk_data = chunk.strip()
                        if chunk_data.startswith("data: "):
                            chunk_content = chunk_data[6:]
                            if chunk_content == "[DONE]":
                                break
                            try:
                                content = json.loads(chunk_content)
                                if "content" in content:
                                    bot_message += content["content"]
                                    update_chat_display(bot_message + "▌")
                                elif "error" in content:
                                    st.error(f"Error: {content['error']}")
                                    break
                            except json.JSONDecodeError:
                                st.error(f"Invalid JSON: {chunk_content}")
                                break

        # 최종 메시지 저장
        st.session_state.messages.append(
            {"role": "assistant", "content": f"{bot_message} ({current_time})"}
        )

        st.session_state.last_input = st.session_state.chat_input
        st.session_state.chat_input = ""
        update_chat_display()


def update_chat_display(current_bot_message=None):
    chat_content = ""
    for message in st.session_state.messages:
        chat_content += display_message(message["role"], message["content"])
    if current_bot_message:
        chat_content += display_message("assistant", current_bot_message)
    chat_container.markdown(
        f'<div class="chat-container">{chat_content}</div>', unsafe_allow_html=True
    )


# 유튜브 URL 입력 받기
url = st.text_input("유튜브 URL을 입력하세요:", key="youtube_url")

# URL 입력 및 스크립트 추출을 위한 버튼 클릭 상태 확인
if st.button("스크립트 추출"):
    if url:
        if "youtu" not in url:
            st.warning("유효한 유튜브 URL을 입력하세요.")
        else:
            headers = {
                "x-session-id": st.session_state.session_id
            }  # 세션 ID 헤더에 포함
            # API 호출 결과를 st.session_state에 저장하여 리렌더링 없이 데이터를 유지하도록 함
            response = requests.get(
                "http://0.0.0.0:8080/get_title_hash",
                params={"url": url},
                headers=headers,
            )
            if response.status_code == 200:
                data = response.json()
                st.session_state.title = data.get("title", "제목")
                st.session_state.hashtags = data.get("hashtags", "")
                st.session_state.video_id = (
                    url.split("v=")[-1] if "v=" in url else url.split("/")[-1]
                )

                with st.spinner("요약 중 입니다."):
                    response = requests.get(
                        "http://0.0.0.0:8080/get_script_summary",
                        params={"url": url},
                        headers=headers,
                    ).json()
                    st.session_state.summary = response.get(
                        "summary_result", "요약 내용이 없습니다."
                    )
                    st.session_state.language = response.get("language")

                    st.session_state.transcript = response.get("script", [])

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
                for item in st.session_state.transcript:
                    st.write(f"{item['start']}초 - {item['end']}초: {item['text']}")

    with col2:
        st.subheader("AI 채팅")

        chat_container = st.empty()
        update_chat_display()

        chat_input = st.text_input(
            "메시지를 입력하세요", key="chat_input", on_change=process_input
        )


# 스크롤 함수 호출 (필요한 경우)
st.markdown("<script>scrollToBottom();</script>", unsafe_allow_html=True)

# 'Enter' 키 처리 및 자동 스크롤을 위한 JavaScript
st.markdown(
    """
<script>
const inputElement = window.parent.document.querySelector('.stTextInput input');
inputElement.addEventListener('keydown', function(e) {
    if (e.key === 'Enter') {
        setTimeout(() => {
            const submitButton = window.parent.document.querySelector('button[kind="primaryFormSubmit"]');
            if (submitButton) {
                submitButton.click();
            }
        }, 10);
    }
});

function scrollChatToBottom() {
    const chatContainer = window.parent.document.querySelector('.chat-container');
    if (chatContainer) {
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }
}

// 새 메시지가 추가될 때마다 스크롤
function observeChatChanges() {
    const chatContainer = window.parent.document.querySelector('.chat-container');
    if (chatContainer) {
        const observer = new MutationObserver(scrollChatToBottom);
        observer.observe(chatContainer, { childList: true, subtree: true });
    }
}

// 페이지 로드 시 초기 설정
document.addEventListener('DOMContentLoaded', function() {
    scrollChatToBottom();
    observeChatChanges();
});
</script>
""",
    unsafe_allow_html=True,
)
