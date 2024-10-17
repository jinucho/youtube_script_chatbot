# 기본 이미지 설정
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# 작업 디렉토리 설정
WORKDIR /app

# 필요한 시스템 패키지 설치 (Python, Java, Git, Node.js)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    python3-pip \
    openjdk-17-jdk \
    libatlas-base-dev \
    g++ \
    git \
    curl && \
    curl -fsSL https://deb.nodesource.com/setup_18.x | bash - && \
    apt-get install -y nodejs && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Python 패키지 의존성 파일 복사 및 설치
RUN pip3 install --force-reinstall "faster-whisper @ https://github.com/SYSTRAN/faster-whisper/archive/refs/heads/master.tar.gz"
COPY requirements.txt requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

# youtube-po-token-generator 클론 및 의존성 설치
RUN git clone https://github.com/YunzheZJU/youtube-po-token-generator.git && \
    cd youtube-po-token-generator && \
    npm install

# whisper 모델 다운로드
RUN python3 -c "from faster_whisper import WhisperModel; model=WhisperModel('large-v3')"

# 애플리케이션 코드 복사
COPY . .

# runpod_wrapper.py 실행
CMD ["python3", "runpod_main.py"]
