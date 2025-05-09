# # 기본 이미지 설정
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# 작업 디렉토리 설정
WORKDIR /app

# 환경 변수 설정
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/app/venv/bin:$PATH"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"
ENV CUDA_HOME="/usr/local/cuda"

# NVIDIA 저장소 및 키 설정

RUN apt-get clean && rm -rf /var/lib/apt/lists/* && apt-get update

RUN rm /etc/apt/sources.list.d/cuda-ubuntu2204-x86_64.list && \
    apt-get update && apt-get install -y --no-install-recommends \
    wget \
    gnupg2 \
    software-properties-common && \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin && \
    mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600 && \
    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub && \
    add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /"

# 필수 패키지 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    python3-pip \
    python3-venv \
    openjdk-17-jdk \
    libatlas-base-dev \
    g++ \
    git \
    curl \
    ffmpeg \
    cuda-libraries-12-1 \
    cuda-libraries-dev-12-1 \
    libcublas-12-1 \
    libcublas-dev-12-1 \
    --allow-change-held-packages \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Python 가상 환경 설정
RUN python3 -m venv /app/venv
ENV VIRTUAL_ENV=/app/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# pip 업그레이드 및 기본 패키지 설치
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# # PyTorch 및 CUDA 12.1 설치
RUN pip install torch==2.4.1+cu121 torchaudio==2.4.1+cu121 --index-url https://download.pytorch.org/whl/cu121

# 의존성 설치
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 다운로드 디렉토리 생성 및 권한 설정
RUN mkdir -p /app/downloads && chmod -R 755 /app/downloads

# 애플리케이션 코드 복사
COPY . .

# Whisper 모델 다운로드 및 초기화
RUN python3 -c "from faster_whisper import WhisperModel; model = WhisperModel('large-v3')"
RUN python3 -c "from langchain_huggingface import HuggingFaceEmbeddings; model = HuggingFaceEmbeddings(model_name='BAAI/bge-m3')"

# 실행 명령
CMD ["python", "runpod_main.py"]
