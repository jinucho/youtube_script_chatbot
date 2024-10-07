from pytube import YouTube
from moviepy.editor import *
import speech_recognition as sr
import cv2
import os


# 유튜브 영상 다운로드 함수
def download_video(url):
    yt = YouTube(url)
    video = yt.streams.filter(file_extension="mp4").first()
    video.download(filename="downloaded_video.mp4")
    return "downloaded_video.mp4"


# 영상에서 음성 추출
def extract_audio(video_path):
    video = VideoFileClip(video_path)
    audio = video.audio
    audio.write_audiofile("extracted_audio.wav")
    return "extracted_audio.wav"


# 음성을 텍스트로 변환
def transcribe_audio(audio_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio_data = recognizer.record(source)
        text = recognizer.recognize_google(audio_data)
        return text


# 영상을 프레임 단위로 분할 및 프레임간 이미지의 차이 계산
import cv2

cap = cv2.VideoCapture("vancouver2.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)

timestamps = [cap.get(cv2.CAP_PROP_POS_MSEC)]
calc_timestamps = [0.0]

while cap.isOpened():
    frame_exists, curr_frame = cap.read()
    if frame_exists:
        timestamps.append(cap.get(cv2.CAP_PROP_POS_MSEC))
        calc_timestamps.append(calc_timestamps[-1] + 1000 / fps)
    else:
        break

cap.release()

for i, (ts, cts) in enumerate(zip(timestamps, calc_timestamps)):
    print("Frame %d difference:" % i, abs(ts - cts))


from langchain import OpenAI, TextSplitter, Document, DocumentLoader
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import MapReduceChain
from langchain.memory import ConversationBufferMemory

# OpenAI API 설정
llm = OpenAI(temperature=0.7, openai_api_key="your_openai_api_key")

# 1. 긴 텍스트를 읽어와서 Document로 로드
document_text = """
여기에 매우 긴 텍스트를 입력합니다. 예를 들어 동영상의 음성 데이터를 텍스트로 변환한 결과가 여기에 들어갑니다. 
이 텍스트는 매우 길고 여러 소주제를 포함하고 있으며, 우리는 이를 효과적으로 분할하고 요약해야 합니다.
"""

# 2. 텍스트를 Document로 변환
documents = [Document(page_content=document_text)]

# 3. TextSplitter를 이용하여 텍스트를 중첩된 형태로 분할
# 각 분할 단위는 1000자로 설정하되, 다음 텍스트가 이전 텍스트의 마지막 200자를 포함하도록 설정
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
split_docs = splitter.split_documents(documents)

# 4. 요약을 위한 체인 로드
summarize_chain = load_summarize_chain(llm, chain_type="map_reduce")

# 5. 분할된 문서 각각에 대해 요약 수행
initial_summary = summarize_chain.run(split_docs)

# 6. 요약된 내용을 다시 하나의 문서로 합침
final_doc = Document(page_content=initial_summary)

# 7. 최종 요약 수행
final_summary = summarize_chain.run([final_doc])

# 결과 출력
print("초기 요약:", initial_summary)
print("최종 요약:", final_summary)


from langchain import OpenAI, LLMChain
from langchain.chains import SimpleSequentialChain
from langchain.agents import Tool, initialize_agent, AgentType
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter

# LLM 초기화 (예: OpenAI GPT-4)
llm = OpenAI(temperature=0.7, openai_api_key="your_openai_api_key")

# 긴 텍스트 예제 (동영상 음성 데이터로 변환된 텍스트)
document_text = """
긴 텍스트가 여기에 들어갑니다. 전체 주제는 "M&A 산업 동향"이며, 소주제는 다음과 같이 나눌 수 있습니다.
1. M&A의 정의와 개요
2. 최근 M&A 시장 동향
3. M&A 과정에서 발생하는 주요 이슈
4. 성공적인 M&A 사례
"""

# 1. 텍스트를 중첩된 형태로 분할하기
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
split_docs = splitter.create_documents([document_text])

# 2. 각 분할된 텍스트에 대해 전체 주제 및 소주제 요약 생성
summary_prompt_template = """
텍스트를 읽고 전체 주제와 소주제를 요약하세요.
텍스트: {text}
"""

summary_prompt = PromptTemplate(
    input_variables=["text"], template=summary_prompt_template
)
summary_chain = LLMChain(llm=llm, prompt=summary_prompt)

# 각 텍스트 조각 요약 생성
summaries = [summary_chain.run(text=doc.page_content) for doc in split_docs]

# 전체 주제와 소주제 요약본 저장
overall_summary = "M&A 산업 동향에 대한 개요 및 각 소주제에 대한 요약입니다."
subtopics = {
    "M&A의 정의와 개요": summaries[0],
    "최근 M&A 시장 동향": summaries[1],
    "M&A 과정에서 발생하는 주요 이슈": summaries[2],
    "성공적인 M&A 사례": summaries[3],
}


# 3. 질의에 따른 에이전트 설정
# 전체 주제 요약 에이전트
def overall_summary_agent(query):
    return f"전체 주제 요약: {overall_summary}"


# 소주제 요약 에이전트
def subtopic_agent(query):
    for subtopic, content in subtopics.items():
        if subtopic in query:
            return f"질문한 소주제: {subtopic}\n소주제 요약: {content}"
    return "해당 소주제에 대한 정보를 찾을 수 없습니다."


# 4. LangChain Tool 설정
tools = [
    Tool(
        name="Overall Summary Agent",
        func=overall_summary_agent,
        description="전체 주제에 대해 요약된 답변을 제공합니다.",
    ),
    Tool(
        name="Subtopic Agent",
        func=subtopic_agent,
        description="질문한 소주제에 대한 답변을 제공합니다.",
    ),
]

# 5. 에이전트 초기화
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    memory=ConversationBufferMemory(),
)

# 6. 사용자 질의 처리
query_1 = "M&A 산업의 전체적인 동향에 대해 알려주세요."
query_2 = "M&A 과정에서 발생하는 주요 이슈에 대해 설명해 주세요."

response_1 = agent.run(query_1)  # 전체 주제 요약
response_2 = agent.run(query_2)  # 특정 소주제에 대한 답변

# 결과 출력
print("전체 주제 질문에 대한 답변:", response_1)
print("소주제 질문에 대한 답변:", response_2)
