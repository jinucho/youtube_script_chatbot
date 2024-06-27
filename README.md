# youtube_summary_chatbot

## 계획

1. 유튜브 영상 다운로드 함수
2. 영상에서 음성 추출
3. 음성을 텍스트로 변환 -> 텍스트를 RAG용 문서로 사용
4. time stamp 추출
5. 영상을 프레임 단위로 분할 및 프레임간 이미지의 차이 계산 -> time stamp마다 주요 이미지 선정
6. LLM과 3번의 텍스트와 연결(langchain) 
