import streamlit as st
import requests

# Streamlit 웹 애플리케이션 설정
st.set_page_config(layout="wide")  # 전체 레이아웃을 넓게 설정
st.title("유튜브 요약 챗봇")

# CSS 스타일을 사용하여 채팅창 레이아웃 설정
st.markdown(
    """
    <style>
    /* 채팅 내역 상자가 항상 위로 정렬되도록 설정 */
    .chat-container {
        display: flex;
        flex-direction: column-reverse; /* 위로 쌓이도록 설정 */
        height: 400px;
        overflow-y: auto; /* 세로 스크롤 활성화 */
        border: 1px solid #ddd; /* 경계선 추가 */
        padding: 10px;
        margin-bottom: 20px;
    }

    /* 채팅 입력창을 아래에 고정 */
    .chat-input-container {
        position: fixed;
        bottom: 0;
        width: calc(100% - 20px); /* 전체 너비 설정 */
        background-color: #f9f9f9; /* 배경색 설정 */
        padding: 10px;
        border-top: 1px solid #ddd; /* 상단 경계선 추가 */
    }

    /* 페이지 전체 배경 색 설정 */
    body {
        background-color: #212121;
        color: #ffffff;
    }

    /* 텍스트 및 입력창 스타일 변경 */
    input, textarea, button {
        background-color: #333;
        color: #fff;
        border: 1px solid #555;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# 유튜브 URL 입력 받기
url = st.text_input("유튜브 URL을 입력하세요:", key="youtube_url")

# API 엔드포인트
api_endpoint = "http://127.0.0.1:8010/extract_info"

# URL이 입력되었고, 버튼이 클릭되었을 때 동작
if st.button("정보 추출"):
    if url:
        # URL 유효성 확인 (간단한 확인: 'youtube' 문자열이 포함되어 있는지 확인)
        if "youtu" not in url:
            st.warning("유효한 유튜브 URL을 입력하세요.")
        else:
            with st.spinner("정리중 입니다~! 잠시만 기다려주세요~!"):
                response = requests.get(api_endpoint, params={"url": url})

                if response.status_code == 200:
                    # JSON 응답 파싱
                    data = response.json()

                    # Streamlit의 columns 레이아웃 사용
                    col1, col2 = st.columns(2)

                    # 왼쪽에는 스크립트 표시
                    with col1:
                        # 유튜브 비디오 삽입
                        st.subheader(
                            f"제목 : {data.get('title_hash').get('title','제목')}"
                        )
                        st.write(f"{data.get('title_hash').get('hashtags','')}")
                        video_id = (
                            url.split("v=")[-1] if "v=" in url else url.split("/")[-1]
                        )

                        # iframe 태그를 사용하여 유튜브 비디오 임베드
                        st.markdown(
                            f'<iframe width="100%" height="400" src="https://www.youtube.com/embed/{video_id}" '
                            f'frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" '
                            f"allowfullscreen></iframe>",
                            unsafe_allow_html=True,
                        )

                        # 요약 내용 표시
                        st.subheader("요약내용")
                        st.write("요약된 내용들을 나열..")

                        # 스크립트 표시를 위해 expander 사용
                        with st.expander("스크립트 보기"):
                            transcript = data.get("script", [])
                            if transcript:
                                # 자막 정보 출력
                                for item in transcript:
                                    st.write(
                                        f"{item['start']}초 - {item['end']}초: {item['text']}"
                                    )
                            else:
                                st.write("자막 또는 음성 인식 결과가 없습니다.")

                    # 오른쪽에는 채팅창 표시
                    with col2:
                        st.subheader("챗봇")

                        # 채팅 기록을 담을 리스트
                        if "chat_history" not in st.session_state:
                            st.session_state.chat_history = []

                        # 채팅 내역 표시 영역
                        st.markdown(
                            '<div class="chat-container">', unsafe_allow_html=True
                        )
                        for chat in reversed(st.session_state.chat_history):
                            if "user" in chat:
                                st.write(f"**사용자**: {chat['user']}")
                            elif "bot" in chat:
                                st.write(f"**챗봇**: {chat['bot']}")
                        st.markdown("</div>", unsafe_allow_html=True)

                        # 채팅 입력창
                        user_message = st.text_input(
                            "메시지를 입력하세요:", key="chat_input"
                        )
                        if st.button("전송") and user_message:
                            # 사용자가 입력한 메시지를 채팅 기록에 추가
                            st.session_state.chat_history.append({"user": user_message})

                            # 예제 챗봇 응답 - 실제로는 챗봇 엔진을 연결하여 응답을 생성해야 함
                            bot_response = (
                                f"챗봇 응답: {user_message}에 대한 답변입니다."
                            )
                            st.session_state.chat_history.append({"bot": bot_response})

                            # 화면 갱신을 위해 전체 컴포넌트를 리렌더링
                            st.experimental_rerun()
