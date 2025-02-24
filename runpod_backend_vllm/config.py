from pydantic_settings import BaseSettings
from pydantic import BaseModel, Field
import torch
from dotenv import load_dotenv
import os
import json

load_dotenv()

# VOLUME_PATH = "/runpod-volume"
VOLUME_PATH = ""
DATA_PATH = os.path.join(VOLUME_PATH, "data")


from pydantic import BaseModel, Field
from typing import List

class Summary(BaseModel):
    emoji: str = Field(..., description="요약에 사용하는 이모지")
    content: str = Field(..., description="요약된 내용")

class FinalSummary(BaseModel):
    key_topic: str = Field(..., description="주요 주제 내용")
    summaries: List[Summary] = Field(..., description="요약된 내용 리스트")

class RecommendQuestions(BaseModel):
    questions: List[str] = Field(..., description="추천 질문 리스트")

class FullStructure(BaseModel):
    FINAL_SUMMARY: FinalSummary = Field(..., description="최종 요약 정보")
    RECOMMEND_QUESTIONS: RecommendQuestions = Field(..., description="추천 질문 리스트")
    

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

    PARTIAL_SUMMARY_PROMPT_TEMPLATE: str = """
                                            다음 REQUEST에 따라 CONTEXT를 요약하고, 출력은 아래에 제공된 출력 형식(OUTPUT_FORMAT)과 정확히 동일하게 한 번만 작성해주세요.
                                            이 작업은 부분 요약입니다, 너무 많이 요약하지 마세요.
    
                                            REQUEST:
                                            1. 주요 내용을 한국어로 요약하세요.
                                            2. 기술, 전문 용어는 번역하지마세요.
                                            3. 내용과 관계없는 불필요한 것은 추가하지 마세요.
                                            
                                            CONTEXT:
                                            {context}
                                            
                                            OUTPUT_FORMAT(JSON 형식):
                                            {{
                                            "PARTIAL_SUMMARY":["요약된 내용1",
                                                                "요약된 내용2",
                                                                "요약된 내용3",
                                                                ...추가 요약 내용 나열]
                                            }}
                                            OUTPUT:
                                            """
    FINAL_SUMMARY_PROMPT_TEMPLATE: str = """
                                        다음 REQUEST에 따라 CONTEXT를 요약하고, 출력은 아래에 제공된 출력 형식(OUTPUT_FORMAT)과 정확히 동일하게 한 번만 작성해주세요.

                                        REQUEST:
                                        1. 주어진 OUTPUT(JSON 형식) 외의 텍스트나 설명을 포함하지 마세요.
                                        2. CONTEXT와 HUMAN MESSAGE는 출력하지마세요.
                                        3. 단 하나의 OUTPUT만 출력하세요.
                                        4. 주요 내용을 한국어로 요약하되, 전문, 기술 용어는 원본을 사용하세요.
                                        5. 요약된 각 문장은 해당 의미와 잘 어울리는 이모지 하나로 시작해야 합니다.
                                        6. 다양한 이모지를 사용하여 요약을 흥미롭게 작성하되, 간결하고 관련성 있게 유지하세요.
                                        7. 문서의 단일 주요 주제와 전반적인 요약에만 집중하세요.
                                        8. 각 요약에서 주요 주제를 명확히 나타내세요.
                                        9. CONTEXT의 내용이 충분히 많다면, 요약 문장을 충분히 생성하세요.
                                        10. 요약된 내용을 기반으로 가장 관련성 높은 세 가지 질문을 한국어로 작성하세요.

                                        CONTEXT:
                                        {context}

                                        OUTPUT_FORMAT(JSON 형식):
                                        {{
                                            "FINAL_SUMMARY": {{
                                                "Key_topic": 주요 주제 내용,
                                                "Summaries": [
                                                    "• Emoji 요약된 내용1",
                                                    "• Emoji 요약된 내용2",
                                                    ...추가 요약 내용 나열
                                                ]
                                            }},
                                            "RECOMMEND_QUESTIONS": [
                                                "첫 번째 질문 (한국어)",
                                                "두 번째 질문 (한국어)",
                                                "세 번째 질문 (한국어)"
                                            ]
                                        }}
                                        OUTPUT:
                                        """

    class Config:
        env_file = ".env"
        extra = "ignore"  # 추가 필드 무시


# 환경 변수 로드 및 설정 객체 생성
settings = Settings()
backup_data = BackupData()
