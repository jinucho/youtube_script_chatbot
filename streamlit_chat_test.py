import streamlit as st
from streamlit_float import *

st.set_page_config(layout="wide")

float_init(theme=True, include_unstable_primary=False)


def chat_content():
    user_input = st.session_state.content
    bot_response = f"네, {user_input}"

    st.session_state["contents"].append({"role": "user", "content": user_input})
    st.session_state["contents"].append({"role": "bot", "content": bot_response})
    st.session_state.content = ""  # Clear the input after sending


if "contents" not in st.session_state:
    st.session_state["contents"] = []

# Main content area
main_container = st.container()

# Fixed input area at the bottom
input_container = st.container()

with main_container:
    for message in st.session_state.contents:
        with st.chat_message(
            name=message["role"], avatar="🤖" if message["role"] == "bot" else None
        ):
            st.write(message["content"])

    # Add some space to prevent overlap with input field
    st.markdown("<br>" * 3, unsafe_allow_html=True)

with input_container:
    st.chat_input(key="content", on_submit=chat_content)

# CSS to fix the input container at the bottom
st.markdown(
    """
    <style>
    .stApp {
        margin-bottom: 80px;
    }
    [data-testid="stChatInput"] {
        position: fixed;
        bottom: 0;
        background-color: white;
        z-index: 1000;
        padding: 1rem;
        left: 0;
        right: 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
