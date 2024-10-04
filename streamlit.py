import streamlit as st
import requests

# Streamlit 웹 애플리케이션 설정
st.title("유튜브 영상 정보 추출")

# 유튜브 URL 입력 받기
url = st.text_input("유튜브 URL을 입력하세요:")

# FastAPI 서버의 엔드포인트 URL 설정 (자신의 서버 주소로 변경)
api_endpoint = "http://127.0.0.1:8010/extract_info"

# URL이 입력되었고, 버튼이 클릭되었을 때 동작
if st.button("정보 추출"):
    if url:
        with st.spinner("정리중 입니다~! 잠시만 기다려주세요~!"):
            response = requests.get(api_endpoint, params={"url": url})

            if response.status_code == 200:
                # JSON 응답 파싱
                data = response.json()

                # 응답 데이터 출력
                st.write("### 영상 제목")
                st.write(
                    data.get("title_hash").get("title", "제목을 가져올 수 없습니다.")
                )

                st.write("### 해시태그")
                st.write(
                    data.get("title_hash").get(
                        "hashtags", "해시태그를 가져올 수 없습니다."
                    )
                )

                st.write("### 내용")
                transcript = data.get("script", [])
                if transcript:
                    # 자막 정보 출력
                    for item in transcript:
                        if "end" in item.keys():
                            st.write(
                                f"{item['start']}초 - {item['end']}초: {item['text']}"
                            )
                        else:
                            st.write(
                                f"{item['start']}초 - {item['duration']}초: {item['text']}"
                            )
                else:
                    st.write("자막 또는 음성 인식 결과가 없습니다.")
            else:
                st.write(
                    f"Error: 서버에서 정보를 가져올 수 없습니다. (상태 코드: {response.status_code})"
                )
    else:
        st.write("유튜브 URL을 입력하세요.")
