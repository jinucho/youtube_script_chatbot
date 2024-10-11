# youtube_script_chatbot

## history

### 2024.10.11

#### streamlit, backend
1. 챗봇 응답 실시간 출력
2. RAG 연결 완료(rag_chat 엔드포인트에 docs load, embeddings 다 넣어놔서 응답 느림)

#### ISSUE
1. 전역변수 선언으로 인해, 동시에 서비스를 사용할 경우 스크립트가 마지막 스크립트로 번경 됨
2. whisper 동시처리 불가

### ToDo
1. 전반적인 프로세스 정리
2. 코드 분리
3. 코드 최적화
4. 챗 히스토리 추가

----------------------------------------------------------------------------------------------------------------------------------------

### 2024.10.10
#### 주피터 노트북.
1. whisper stt(번역하지 않고 원문 그대로 추출)
2. 추출된 텍스트 주제 요약(요약은 llm을 통해 번역)
3. 추출된 텍스트 문단 구분
4. 문단을 번역(from deep_translator import GoogleTranslator)
5. 2번에서 요약된 내용을 1번의 원문 docs에 추가
6. 5번의 문서를 FAISS, bm25로 앙상블 리트리벌
7. 6번의 리트리벌을 llm과 연결
8. 결과 확인

#### streamlit, backend
1. Youtube url 입력
2. streamlit에 url로 영상 표시
3. 제목, 해시테그 추출 후 표시
4. 요약 내용 표시
5. 전체 스크립트 표시(영어일 경우 한글로 번역 후 표시)
6. 챗봇 배치용 챗 컨테이너 및 입력창 구현
7. whisper backend에 openai API로 정의된 챗봇과 연결


### ToDo
1. 텍스트 전처리 프로세스 정리 - 기능에 대한 필요 유무 확인 후 제거
2. 챗 히스토리 추가
3. 챗봇 & RAG 구현

----------------------------------------------------------------------------------------------------------------------------------------

### 2024.10.07
1. ~~유튜브 영상 다운로드 함수~~ / 영상은 웹에서 표시만
2. ~~영상에서 음성 추출 / 영상 내 스크립트 추출은 제외~~ / 1차완료
3. ~~음성을 텍스트로 변환~~/ 1차완료
4. 변환 된 텍스트에서 대주제, 소주제 생성 및 소주제 별로 문단 구분
5. 적절하게 전처리된 문단을 RAG용 문서로 사용
6. LLM과 3번의 텍스트와 연결(langchain) 

### ToDo

1. 스크립트 추출 최적화 (whisper 비동기 처리)

----------------------------------------------------------------------------------------------------------------------------------------

### 2024.10.04

1. ~~유튜브 영상 다운로드 함수~~ / 영상은 웹에서 표시만
2. 영상에서 음성 추출 / 영상 내 스크립트 추출은 제외
3. 음성을 텍스트로 변환
4. 변환 된 텍스트에서 대주제, 소주제 생성 및 소주제 별로 문단 구분
5. 적절하게 전처리된 문단을 RAG용 문서로 사용
6. LLM과 3번의 텍스트와 연결(langchain) 

----------------------------------------------------------------------------------------------------------------------------------------

### 2024.06.28

1. 유튜브 영상 다운로드 함수
2. 영상에서 음성 추출
3. 음성을 텍스트로 변환 -> 텍스트를 RAG용 문서로 사용
4. 영상을 프레임 단위로 분할 및 프레임간 이미지의 차이 계산 -> 프레임간 이미지의 차이값을 통해 영상의 소주제를 구분/ 소주제마다의 key frame 이미지를 몇가지 저장
5. LLM과 3번의 텍스트와 연결(langchain) / 챗봇으로 사용하기 위한 finetunning방법 조사 후 적용
4. time stamp 추출
5. 영상을 프레임 단위로 분할 및 프레임간 이미지의 차이 계산 -> time stamp마다 주요 이미지 선정
6. LLM과 3번의 텍스트와 연결(langchain)

----------------------------------------------------------------------------------------------------------------------------------------

### 2024.06.27

1. 유튜브 영상 다운로드 함수
2. 영상에서 음성 추출
3. 음성을 텍스트로 변환 -> 텍스트를 RAG용 문서로 사용
4. 영상을 프레임 단위로 분할 및 프레임간 이미지의 차이 계산 -> 프레임간 이미지의 차이값을 통해 영상의 소주제를 구분/ 소주제마다의 key frame 이미지를 몇가지 저장
5. LLM과 3번의 텍스트와 연결(langchain) / 챗봇으로 사용하기 위한 finetunning방법 조사 후 적용

----------------------------------------------------------------------------------------------------------------------------------------

### 2024.06.20 사용할 함수 기록
```python
from pytube import YouTube
from moviepy.editor import *
import speech_recognition as sr
import cv2
import os
#유튜브 영상 다운로드 함수
def download_video(url):
    yt = YouTube(url)
    video = yt.streams.filter(file_extension='mp4').first()
    video.download(filename='downloaded_video.mp4')
    return 'downloaded_video.mp4'
#영상에서 음성 추출
def extract_audio(video_path):
    video = VideoFileClip(video_path)
    audio = video.audio
    audio.write_audiofile('extracted_audio.wav')
    return 'extracted_audio.wav'
#음성을 텍스트로 변환
def transcribe_audio(audio_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio_data = recognizer.record(source)
        text = recognizer.recognize_google(audio_data)
        return text
# 영상을 프레임 단위로 분할 및 프레임간 이미지의 차이 계산
import cv2
cap = cv2.VideoCapture('vancouver2.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)
timestamps = [cap.get(cv2.CAP_PROP_POS_MSEC)]
calc_timestamps = [0.0]
while(cap.isOpened()):
frame_exists, curr_frame = cap.read()
if frame_exists:
  timestamps.append(cap.get(cv2.CAP_PROP_POS_MSEC))
  calc_timestamps.append(calc_timestamps[-1] + 1000/fps)
else:
  break
cap.release()
for i, (ts, cts) in enumerate(zip(timestamps, calc_timestamps)):
  print('Frame %d difference:'%i, abs(ts - cts))
```

