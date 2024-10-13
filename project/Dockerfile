# # 기본 이미지 설정
# FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# # 작업 디렉토리 설정
# WORKDIR /app

# # 필요한 시스템 패키지 설치
# RUN apt-get update && \
#     apt-get install -y --no-install-recommends \
#     build-essential \
#     python3-dev \  
#     openjdk-17-jdk \ 
#     libatlas-base-dev \
#     g++ && \ 
#     apt-get clean && rm -rf /var/lib/apt/lists/*


# # Java 및 Python 환경 변수 설정
# ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64/ \
#     PATH=$JAVA_HOME/bin:/usr/local/cuda/bin:$PATH \
#     LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$JAVA_HOME/lib/server:/usr/local/cuda/lib64 \
#     PYTHONUNBUFFERED=1

# # Python 패키지 의존성 파일 복사 및 설치
# COPY requirements.txt requirements.txt
# RUN pip install --no-cache-dir -r requirements.txt


# # whisper 모델 다운로드
# RUN python3 -c "from faster_whisper import WhisperModel; model=WhisperModel('large-v3')"

# # 애플리케이션 코드 복사
# COPY . .

# # FastAPI 애플리케이션 실행
# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]

# 기본 이미지 설정 (Ubuntu 22.04로 변경)
FROM ubuntu:22.04

# 작업 디렉토리 설정
WORKDIR /app

# 필요한 시스템 패키지 설치
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    python3 \
    python3-pip \
    python3-dev \
    openjdk-17-jdk \
    libatlas-base-dev \
    g++ && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Java 및 Python 환경 변수 설정 (수정됨)
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH=$JAVA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$JAVA_HOME/lib/server
ENV PYTHONUNBUFFERED=1

# Python 패키지 의존성 파일 복사 및 설치
COPY requirements.txt requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

# whisper 모델 다운로드
RUN python3 -c "from faster_whisper import WhisperModel; model=WhisperModel('large-v3', device='cpu')"

# 애플리케이션 코드 복사
COPY . .

# FastAPI 애플리케이션 실행
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]