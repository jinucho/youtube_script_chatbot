import streamlit as st
import requests
from datetime import datetime, timedelta, timezone


# 한국 표준시(KST) 시간대 설정
kst = timezone(timedelta(hours=9))

# Streamlit 웹 애플리케이션 설정
st.set_page_config(layout="wide")  # 전체 레이아웃을 넓게 설정
st.title("유튜브 요약 및 AI 채팅")


# 초기 상태 설정: st.session_state를 사용하여 상태 초기화
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # 채팅 기록 초기화

if "user_input_field" not in st.session_state:
    st.session_state.user_input_field = ""

if "title" not in st.session_state:
    st.session_state.title = ""
    st.session_state.hashtags = ""
    st.session_state.video_id = ""
    st.session_state.summary = ""
    st.session_state.language = ""
    st.session_state.transcript = []

if "transcript_expanded" not in st.session_state:
    st.session_state.transcript_expanded = False

# 유튜브 URL 입력 받기
url = st.text_input("유튜브 URL을 입력하세요:", key="youtube_url")

# URL 입력 및 스크립트 추출을 위한 버튼 클릭 상태 확인
if st.button("스크립트 추출"):
    if url:
        # URL 유효성 확인 (간단한 확인: 'youtube' 문자열이 포함되어 있는지 확인)
        if "youtu" not in url:
            st.warning("유효한 유튜브 URL을 입력하세요.")
        else:
            # API 호출 결과를 st.session_state에 저장하여 리렌더링 없이 데이터를 유지하도록 함
            response = requests.get(
                "http://127.0.0.1:8010/get_title_hash", params={"url": url}
            )
            if response.status_code == 200:
                # JSON 응답 파싱 및 st.session_state에 데이터 저장
                data = response.json()
                st.session_state.title = data.get("title", "제목")
                st.session_state.hashtags = data.get("hashtags", "")
                st.session_state.video_id = (
                    url.split("v=")[-1] if "v=" in url else url.split("/")[-1]
                )

                with st.spinner("요약 중 입니다."):
                    summary_response = requests.get(
                        "http://127.0.0.1:8010/get_script_summary",
                        params={"url": url},
                    ).json()
                    st.session_state.summary = summary_response.get(
                        "summary_result", "요약 내용이 없습니다."
                    )
                    st.session_state.language = summary_response.get("language")

                with st.spinner("전체 스크립트"):
                    script_response = requests.get(
                        "http://127.0.0.1:8010/get_script",
                        params={"language": st.session_state.language},
                    ).json()
                    st.session_state.transcript = script_response.get("script", [])

# URL이 입력되었고, 데이터가 session_state에 저장된 경우 표시
if st.session_state.title:
    col1, col2 = st.columns(2)

    # 왼쪽에는 스크립트 표시
    with col1:
        # 유튜브 비디오 삽입
        st.subheader(f"제목 : {st.session_state.title}")
        st.write(st.session_state.hashtags)

        # iframe 태그를 사용하여 유튜브 비디오 임베드
        st.markdown(
            f'<iframe width="100%" height="600" src="https://www.youtube.com/embed/{st.session_state.video_id}" '
            f'frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" '
            f"allowfullscreen></iframe>",
            unsafe_allow_html=True,
        )

        # 요약 내용 표시
        st.subheader("요약내용")
        st.write(st.session_state.summary)

        with st.expander(
            "스크립트 보기", expanded=st.session_state.transcript_expanded
        ):
            if st.session_state.transcript:
                with st.container(height=400):
                    # 자막 정보 출력
                    for item in st.session_state.transcript:
                        st.write(f"{item['start']}초 - {item['end']}초: {item['text']}")

                # 컨테이너 상태 업데이트
                st.session_state.transcript_expanded = True
            else:
                st.write("자막 또는 음성 인식 결과가 없습니다.")

    # 오른쪽 컬럼 (채팅창) 부분
    with col2:
        st.subheader("AI 채팅")

        # CSS, 채팅 내역, JavaScript를 모두 하나의 HTML로 통합
        chat_html = f"""
        <style>
        .chat-container {{
            display: flex;
            flex-direction: column;
            height: calc(100vh - 50px);
            overflow-y: auto;
            border: 1px solid #ddd;
            padding: 10px;
            padding-bottom: 10px;
            margin-bottom: 50px;
            background-color: #f7f7f7;
        }}

        .user-message {{
            background-color: #1AB5D5;
            color: #000;
            border-radius: 10px;
            padding: 8px;
            margin: 5px;
            align-self: flex-end;
            max-width: 60%;
            word-wrap: break-word;
        }}

        .bot-message {{
            background-color: #ECECEC;
            color: #000;
            border-radius: 10px;
            padding: 8px;
            margin: 5px;
            align-self: flex-start;
            max-width: 60%;
            word-wrap: break-word;
        }}

        .chat-input {{
            width: calc(100% - 20px);
            padding: 10px;
            margin-top: 10px;
        }}
        </style>

        <div class="chat-container" id="chat-container">
            {''.join([
                f'<div class="user-message">{chat["user"]}</div>' if "user" in chat 
                else f'<div class="bot-message">{chat["bot"]}</div>'
                for chat in st.session_state.chat_history
            ])}
        </div>

        <script>
            function scrollToBottom() {{
                const chatContainer = document.getElementById('chat-container');
                if (chatContainer) {{
                    chatContainer.scrollTop = chatContainer.scrollHeight+200;
                }}
            }}

            // 초기 스크롤
            scrollToBottom();
            
            // DOM 변경 감지를 위한 MutationObserver 설정
            const observer = new MutationObserver(scrollToBottom);
            const chatContainer = document.getElementById('chat-container');
            if (chatContainer) {{
                observer.observe(chatContainer, {{ 
                    childList: true, 
                    subtree: true 
                }});
            }}
        </script>
        """

        # components.v1.html을 사용하여 통합된 HTML 실행
        st.components.v1.html(chat_html, height=500)

        # 입력 폼
        with st.form(key="chat_form", clear_on_submit=True):
            user_input = st.text_input("메시지를 입력하세요:", key="user_input_field")
            submit_button = st.form_submit_button("전송")

            if submit_button and user_input.strip():
                current_time = datetime.now(kst).strftime("%H:%M")

                # 사용자 메시지 추가
                st.session_state.chat_history.append(
                    {"user": f"{user_input} ({current_time})"}
                )

                # 봇 응답 생성 및 추가
                bot_response = requests.get(
                    "http://127.0.0.1:8010/chat",
                    params={"prompt": user_input},
                ).json()

                st.session_state.chat_history.append(
                    {"bot": f"{bot_response.get('result')} ({current_time})"}
                )

                st.rerun()
