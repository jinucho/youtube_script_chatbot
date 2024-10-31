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

    PARTIAL_SUMMARY_PROMPT_TEMPLATE: str = """Please summarize the sentence according to the following REQUEST.
    
                                            REQUEST:
                                            1. Summarize the main points in bullet points in KOREAN.
                                            2. Each summarized sentence must start with an emoji that fits the meaning of the each sentence.
                                            3. Use various emojis to make the summary more interesting.
                                            4. Translate the summary into KOREAN if it is written in ENGLISH.
                                            5. DO NOT translate any technical terms.\n6. DO NOT include any unnecessary information.
                                            
                                            CONTEXT:
                                            {context}
                                            
                                            SUMMARY:"
                                            """
    FINAL_SUMMARY_PROMPT_TEMPLATE: str = """Please summarize the sentence according to the following FINAL REQUEST and provide the output EXACTLY as shown in the example format below. Do not modify the section headers or format in any way.

                                            FINAL REQUEST:
                                            1. The provided summary sections are partial summaries of one document. Please combine them into a single cohesive summary.
                                            2. Summarize the main points in bullet points in KOREAN, but DO NOT translate any technical terms.
                                            3. Each summarized sentence must start with a single emoji that fits the meaning of the sentence.
                                            4. Use various emojis to make the summary more interesting, but keep it concise and relevant.
                                            5. Focus on identifying and presenting only one main topic and one overall summary for the document.
                                            6. Avoid redundant or repeated points, and ensure that the summary covers all key ideas without introducing multiple conclusions or topics.
                                            7. Please refer to each summary and indicate the key topic.
                                            8. If the original text is in English, we have already provided a summary translated into Korean, so please do not provide a separate translation.
                                            9. Based on the summarized content, please create three recommended questions.

                                            CONTEXT:
                                            {context}

                                            YOUR RESPONSE MUST FOLLOW THIS EXACT FORMAT:

                                            [FINAL SUMMARY]
                                            Key topic: [Key topic]
                                            • 🎯 First summary point\n
                                            • 📚 Second summary point\n
                                            • 💡 Third summary point\n
                                            • ...

                                            [RECOMMEND QUESTIONS]
                                            1. First question\n
                                            2. Second question\n
                                            3. Third question

                                            IMPORTANT FORMATTING RULES:
                                            - Use EXACTLY '[FINAL SUMMARY]' and '[RECOMMEND QUESTIONS]' as section headers
                                            - Start each summary point with '• ' followed by an emoji
                                            - Number questions with '1. ', '2. ', '3. '
                                            - Do not add any additional headers or sections
                                            - Do not modify the format of the section headers
                                            - Leave exactly one blank line between sections"""

    class Config:
        env_file = ".env"
        extra = "ignore"  # 추가 필드 무시


# 환경 변수 로드 및 설정 객체 생성
settings = Settings()
