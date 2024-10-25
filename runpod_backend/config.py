from pydantic_settings import BaseSettings
import torch
from dotenv import load_dotenv
import os

load_dotenv()


class Settings(BaseSettings):
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    COMPUTE_TYPE: str = "float16" if DEVICE == "cuda" else "int8"

    # LangChain 관련 설정 추가
    LANGCHAIN_TRACING_V2: bool = os.getenv("LANGCHAIN_TRACING_V2")
    LANGCHAIN_ENDPOINT: str = os.getenv("LANGCHAIN_ENDPOINT")
    LANGCHAIN_API_KEY: str = os.getenv("LANGCHAIN_API_KEY")
    LANGCHAIN_PROJECT: str = os.getenv("LANGCHAIN_PROJECT")

    PARTIAL_SUMMARY_PROMPT_TEMPLATE: str = (
        'Please summarize the sentence according to the following REQUEST.\nREQUEST:\n1. Summarize the main points in bullet points in KOREAN.\n2. Each summarized sentence must start with an emoji that fits the meaning of the each sentence.\n3. Use various emojis to make the summary more interesting.\n4. Translate the summary into KOREAN if it is written in ENGLISH.\n5. DO NOT translate any technical terms.\n6. DO NOT include any unnecessary information.\n\nCONTEXT:\n{context}\n\nSUMMARY:"\n'
    )
    FINAL_SUMMARY_PROMPT_TEMPLATE: str = (
        "Please summarize the sentence according to the following FINAL REQUEST. \nFINAL REQUEST:\n1. The provided summary sections are partial summaries of one document. Please combine them into a single cohesive summary.\n2. Summarize the main points in bullet points in KOREAN, but DO NOT translate any technical terms.\n3. Each summarized sentence must start with a single emoji that fits the meaning of the sentence.\n4. Use various emojis to make the summary more interesting, but keep it concise and relevant.\n5. Focus on identifying and presenting only one main topic and one overall summary for the document.\n6. Avoid redundant or repeated points, and ensure that the summary covers all key ideas without introducing multiple conclusions or topics.\n7. Please refer to each summary and indicate the key topic.\n8. If the original text is in English, we have already provided a summary translated into Korean, so please do not provide a separate translation.\n\nCONTEXT: \n{context}\n\nFINAL SUMMARY:"
    )

    class Config:
        env_file = ".env"
        extra = "ignore"  # 추가 필드 무시


# 환경 변수 로드 및 설정 객체 생성
settings = Settings()
