import streamlit as st
import time

st.title("비대칭 컬럼 레이아웃")

# 세션 상태 초기화
if "is_processed" not in st.session_state:
    st.session_state.is_processed = False

# 2:1 비율로 컬럼 분할
left_column, right_column = st.columns([2, 1])

# 왼쪽 컬럼 상단 (항상 표시)
with left_column:
    st.markdown('<p class="big-font">좌측 상단입니다.</p>', unsafe_allow_html=True)

    # 처리 버튼
    if not st.session_state.is_processed and st.button("처리 시작"):
        with st.spinner("처리 중..."):
            # 시뮬레이션을 위한 반복문
            progress_text = st.empty()
            for i in range(5):
                progress_text.text(f"처리 중... {i+1}/5")
                time.sleep(1)
            st.session_state.is_processed = True
            st.rerun()

    # 처리가 완료된 경우 하단 내용 표시
    if st.session_state.is_processed:
        st.markdown(
            '<p class="big-font">좌측 하단입니다. (처리 완료)</p>',
            unsafe_allow_html=True,
        )

# 오른쪽 컬럼 (처리가 완료된 경우에만 표시)
if st.session_state.is_processed:
    with right_column:
        st.markdown(
            '<p class="big-font">우측입니다. (처리 완료 후 표시)</p>',
            unsafe_allow_html=True,
        )
