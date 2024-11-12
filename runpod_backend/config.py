from pydantic_settings import BaseSettings
import torch
from dotenv import load_dotenv
import os
import json

load_dotenv()

# VOLUME_PATH = "/runpod-volume"
VOLUME_PATH = ""
DATA_PATH = os.path.join(VOLUME_PATH, "data")

class BackupData:
    def __init__(self, file_path=f"{DATA_PATH}/backup.json"):
        self.file_path = file_path
        try:
            with open(self.file_path, "r", encoding="utf-8") as file:
                self.data = json.load(file)
        except FileNotFoundError:
            self.data = {}  # 파일이 없을 경우 빈 딕셔너리로 초기화

    def add_title_and_hashtags(self, url_id, title, hashtags):
        # url_id별로 title과 hashtags를 추가하거나 업데이트
        if url_id not in self.data:
            self.data[url_id] = {}
        self.data[url_id].update({"title": title, "hashtags": hashtags})
        self._save_data()

    def add_data(self, url_id, type, data):
        # url_id별로 audio_url을 따로 추가하거나 업데이트
        if url_id not in self.data:
            self.data[url_id] = {}
        self.data[url_id][type] = data
        self._save_data()

    def get(self, url_id):
        # url_id로 데이터 조회
        return self.data.get(url_id, None)

    def _save_data(self):
        # JSON 파일에 데이터를 저장하는 내부 메서드
        with open(self.file_path, "w", encoding="utf-8") as file:
            json.dump(self.data, file, ensure_ascii=False, indent=4)


def custom_parser(text):
    summary = (
        text.split("[FINAL SUMMARY]")[1].split("[RECOMMEND QUESTIONS]")[0].strip("\n\n")
    )
    questions = text.split("[FINAL SUMMARY]")[1].split("[RECOMMEND QUESTIONS]")[1]
    return summary, questions


class Settings(BaseSettings):
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    COMPUTE_TYPE: str = "float16" if DEVICE == "cuda" else "int8"

    # LangChain 관련 설정 추가
    LANGCHAIN_TRACING_V2: bool = os.getenv("LANGCHAIN_TRACING_V2")
    LANGCHAIN_ENDPOINT: str = os.getenv("LANGCHAIN_ENDPOINT")
    LANGCHAIN_API_KEY: str = os.getenv("LANGCHAIN_API_KEY")
    LANGCHAIN_PROJECT: str = os.getenv("LANGCHAIN_PROJECT")

    MODEL_NAME:str = "BAAI/bge-m3"
    ENCODE_KWARGS:dict = {"normalize_embeddings": True}

    DATA_PATH:str = DATA_PATH

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
                                            2. If the content of the document is sufficient, please ensure the summary includes key details and is at least 10 summary points.
                                            3. Summarize the main points in bullet points in KOREAN, but DO NOT translate any technical terms.
                                            4. Each summarized sentence must start with a single emoji that fits the meaning of the sentence.
                                            5. Use various emojis to make the summary more interesting, but keep it concise and relevant.
                                            6. Focus on identifying and presenting only one main topic and one overall summary for the document.
                                            7. Avoid redundant or repeated points, and ensure that the summary covers all key ideas without introducing multiple conclusions or topics.
                                            8. Please refer to each summary and indicate the key topic.
                                            9. If the original text is in English, we have already provided a summary translated into Korean, so please do not provide a separate translation.
                                            10. Based on the summarized content, please create the three most relevant recommended questions.

                                            CONTEXT:
                                            {context}

                                            YOUR RESPONSE MUST FOLLOW THIS EXACT FORMAT:

                                            [FINAL SUMMARY]
                                            Key topic: [Key topic]\n
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
backup_data = BackupData()
