# 기본 이미지 설정
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# 작업 디렉토리 설정
WORKDIR /app

# 환경 변수 설정
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/app/venv/bin:$PATH"
# ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:/app/venv/lib/python3.10/site-packages/torch/lib:$LD_LIBRARY_PATH"
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

# PyTorch 및 CUDA 12.1 설치 (NVIDIA 공식 버전)
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121


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
RUN python3 -c "from transformers import AutoModelForCausalLM, AutoTokenizer; model = AutoModelForCausalLM.from_pretrained('beomi/Qwen2.5-7B-Instruct-kowiki-qa-context'); tokenizer = tokenizer = AutoTokenizer.from_pretrained('beomi/Qwen2.5-7B-Instruct-kowiki-qa-context')"

# vLLM과 runpod_main.py를 순차적으로 실행하기 위한 스크립트 생성
RUN echo '#!/bin/bash\n\
python -m vllm.entrypoints.openai.api_server --model beomi/Qwen2.5-7B-Instruct-kowiki-qa-context --max-model-len=23000 --gpu-memory-utilization=0.7 --host 0.0.0.0 --port 8000 &\n\
VLLM_PID=$!\n\
echo "vLLM 서버가 시작되었습니다. PID: $VLLM_PID"\n\
# vLLM 서버가 완전히 시작될 때까지 대기\n\
sleep 30\n\
# runpod_main.py 실행\n\
python runpod_main.py\n\
# 스크립트가 종료되면 vLLM 서버도 종료\n\
kill $VLLM_PID\n\
' > /app/start_services.sh && chmod +x /app/start_services.sh

# 시작 스크립트 실행
CMD ["/app/start_services.sh"]


