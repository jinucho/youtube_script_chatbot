# # 기본 이미지 설정
# FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# # 작업 디렉토리 설정
# WORKDIR /app

# # 필요한 시스템 패키지 설치
# RUN apt-get update && \
#     apt-get install -y --no-install-recommends \
#     build-essential \
#     python3-dev \
#     python3-pip \
#     openjdk-17-jdk \
#     libatlas-base-dev \
#     g++ && \
#     apt-get clean && rm -rf /var/lib/apt/lists/*

# # Java 및 Python 환경 변수 설정
# ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
# ENV PATH=$JAVA_HOME/bin:$PATH
# ENV LD_LIBRARY_PATH=$JAVA_HOME/lib/server
# ENV PYTHONUNBUFFERED=1
# ENV KMP_DUPLICATE_LIB_OK=TRUE
# ENV OMP_NUM_THREADS=1

# # Python 패키지 의존성 파일 복사 및 설치
# COPY requirements.txt requirements.txt
# RUN pip3 install --no-cache-dir -r requirements.txt

# # whisper 모델 다운로드
# RUN python3 -c "from faster_whisper import WhisperModel; model=WhisperModel('large-v3')"

# # 애플리케이션 코드 복사
# COPY . .

# # RunPod 환경 변수 설정 (필요한 경우)
# ENV RUNPOD_INTERNAL_API_HOST=http://localhost:8000

# # runpod_wrapper.py 실행
# CMD ["python3", "runpod_main.py"]

# 기본 이미지 설정
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# 작업 디렉토리 설정
WORKDIR /app

# 필요한 시스템 패키지 설치
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    python3-pip \
    openjdk-17-jdk \
    libatlas-base-dev \
    libopenblas-dev \
    g++ && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Java 및 Python 환경 변수 설정
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH=$JAVA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$JAVA_HOME/lib/server
ENV PYTHONUNBUFFERED=1
ENV KMP_DUPLICATE_LIB_OK=TRUE
ENV OMP_NUM_THREADS=1

# pip 업그레이드
RUN pip3 install --upgrade pip

# Python 패키지 의존성 파일 복사 및 설치
COPY requirements.txt requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

# faiss-gpu 설치 (CUDA 11.8 호환 버전)
RUN pip3 install faiss-gpu

# whisper 모델 다운로드
RUN python3 -c "from faster_whisper import WhisperModel; model=WhisperModel('large-v3')"

# 애플리케이션 코드 복사
COPY . .

# RunPod 환경 변수 설정 (필요한 경우)
ENV RUNPOD_INTERNAL_API_HOST=http://localhost:8000

# runpod_wrapper.py 실행
CMD ["python3", "runpod_main.py"]