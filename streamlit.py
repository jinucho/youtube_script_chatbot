import streamlit as st
import requests
from datetime import datetime, timedelta, timezone

# 한국 표준시(KST) 시간대 설정
kst = timezone(timedelta(hours=9))

# Streamlit 웹 애플리케이션 설정
st.set_page_config(layout="wide")  # 전체 레이아웃을 넓게 설정
st.title("유튜브 요약 및 AI 채팅")

# 초기 상태 설정: st.session_state를 사용하여 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

if "title" not in st.session_state:
    st.session_state.title = ""
    st.session_state.hashtags = ""
    st.session_state.video_id = ""
    st.session_state.summary = ""
    st.session_state.language = ""
    st.session_state.transcript = []

if "last_input" not in st.session_state:
    st.session_state.last_input = ""

if "transcript_expanded" not in st.session_state:
    st.session_state.transcript_expanded = False


def display_message(role, content):
    if role == "user":
        st.markdown(
            f"""
            <div style='display: flex; justify-content: flex-end;'>
                <div style='background-color: #DCF8C6; color: black; border-radius: 10px; padding: 8px; margin: 5px; max-width: 70%;'>
                    {content}
                </div>
            </div>
        """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div style='display: flex; justify-content: flex-start;'>
                <div style='background-color: #FFFFFF; color: black; border-radius: 10px; padding: 8px; margin: 5px; max-width: 70%;'>
                    {content}
                </div>
            </div>
        """,
            unsafe_allow_html=True,
        )


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

    with col2:
        st.subheader("AI 채팅")

        # 채팅 컨테이너
        chat_container = st.container(height=600)

        # 채팅창 출력
        with chat_container:
            for message in st.session_state.messages:
                display_message(message["role"], message["content"])

        # 채팅 입력 처리 함수
        def process_input():
            if (
                st.session_state.chat_input
                and st.session_state.chat_input != st.session_state.last_input
            ):
                current_time = datetime.now(kst).strftime("%H:%M")

                # 사용자 메시지 추가
                user_message = f"{st.session_state.chat_input} ({current_time})"
                st.session_state.messages.append(
                    {"role": "user", "content": user_message}
                )

                # 사용자 메시지 즉시 표시
                display_message("user", user_message)

                # 봇 응답 생성 중임을 표시
                with st.spinner("AI가 응답을 생성 중입니다..."):
                    # 봇 응답 생성 및 추가
                    response = requests.get(
                        "http://127.0.0.1:8010/chat",
                        params={"prompt": st.session_state.chat_input},
                    )
                    if response.status_code == 200:
                        bot_response = response.json().get("result", "응답이 없습니다.")
                        bot_message = f"{bot_response} ({current_time})"
                        st.session_state.messages.append(
                            {"role": "assistant", "content": bot_message}
                        )

                        # 봇 메시지 표시
                        display_message("assistant", bot_message)

                # 마지막 입력 업데이트
                st.session_state.last_input = st.session_state.chat_input
                # 입력 필드 초기화
                st.session_state.chat_input = ""

        # 채팅 입력 폼
        chat_input = st.text_input(
            "메시지를 입력하세요", key="chat_input", on_change=process_input
        )

        # 'Enter' 키 처리를 위한 JavaScript
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
            </script>
        """,
            unsafe_allow_html=True,
        )

    # 스크롤을 최하단으로 이동
    st.markdown(
        """
        <script>
            var elements = window.parent.document.querySelectorAll('.stMarkdown');
            var lastElement = elements[elements.length - 1];
            lastElement.scrollIntoView();
        </script>
    """,
        unsafe_allow_html=True,
    )
