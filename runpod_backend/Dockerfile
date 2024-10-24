# # 기본 이미지 설정
# FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# # 작업 디렉토리 설정
# WORKDIR /app


# RUN apt-get update && \
#     apt-get install -y --no-install-recommends \
#     build-essential \
#     python3-dev \
#     python3-pip \
#     openjdk-17-jdk \
#     libatlas-base-dev \
#     g++ \
#     git \
#     curl \
#     ffmpeg && \
#     apt-get clean && rm -rf /var/lib/apt/lists/*



# # Python 패키지 의존성 파일 복사 및 설치
# RUN pip3 install --force-reinstall "faster-whisper @ https://github.com/SYSTRAN/faster-whisper/archive/refs/heads/master.tar.gz"
# COPY requirements.txt requirements.txt
# RUN pip3 install --no-cache-dir -r requirements.txt

# # whisper 모델 다운로드
# RUN python3 -c "from faster_whisper import WhisperModel; model=WhisperModel('large-v3')"

# RUN mkdir -p /app/downloads && chmod -R 755 /app/downloads

# # 애플리케이션 코드 복사
# COPY . .

# # runpod_wrapper.py 실행
# CMD ["python3", "runpod_main.py"]
# FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# # 작업 디렉토리 설정
# WORKDIR /app

# # 환경 변수 설정
# ENV DEBIAN_FRONTEND=noninteractive
# ENV PATH="/app/venv/bin:$PATH"
# ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}"

# # 필수 패키지 설치
# RUN apt-get update --fix-missing && \
#     apt-get install -y --no-install-recommends \
#     build-essential \
#     python3-dev \
#     python3-pip \
#     python3-venv \
#     openjdk-17-jdk \
#     libatlas-base-dev \
#     g++ \
#     git \
#     curl \
#     ffmpeg \
#     software-properties-common \
#     wget \
#     gnupg && \
#     apt-get clean && \
#     rm -rf /var/lib/apt/lists/*

# # Python 가상 환경 설정
# RUN python3 -m venv /app/venv
# ENV VIRTUAL_ENV=/app/venv
# ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# # pip 업그레이드 및 기본 패키지 설치
# RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# # PyTorch 설치 (CUDA 11.8 지원)
# RUN pip install --no-cache-dir torch torchaudio --extra-index-url https://download.pytorch.org/whl/cu118

# # 의존성 설치
# COPY requirements.txt requirements.txt
# RUN pip install --no-cache-dir -r requirements.txt

# # Faster Whisper 설치
# RUN pip install --no-cache-dir "faster-whisper @ https://github.com/SYSTRAN/faster-whisper/archive/refs/heads/master.tar.gz"

# # Whisper 모델 다운로드
# RUN python3 -c "from faster_whisper import WhisperModel; model=WhisperModel('large-v3')"

# # 다운로드 디렉토리 생성 및 권한 설정
# RUN mkdir -p /app/downloads && chmod -R 755 /app/downloads

# # 애플리케이션 코드 복사
# COPY . .

# # 실행 명령
# CMD ["python", "runpod_main.py"]

FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# 작업 디렉토리 설정
WORKDIR /app

# 환경 변수 설정
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/app/venv/bin:$PATH"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"
ENV CUDA_HOME="/usr/local/cuda"

# NVIDIA 저장소 및 키 설정
RUN apt-get update && apt-get install -y --no-install-recommends \
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
    cuda-libraries-11-8 \
    cuda-libraries-dev-11-8 \
    libcublas-12-0 \
    libcublas-dev-12-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Python 가상 환경 설정
RUN python3 -m venv /app/venv
ENV VIRTUAL_ENV=/app/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# pip 업그레이드 및 기본 패키지 설치
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# 의존성 설치
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Faster Whisper 및 관련 라이브러리 설치
RUN pip install --no-cache-dir "faster-whisper @ https://github.com/SYSTRAN/faster-whisper/archive/refs/heads/master.tar.gz"

# 다운로드 디렉토리 생성 및 권한 설정
RUN mkdir -p /app/downloads && chmod -R 755 /app/downloads

# 애플리케이션 코드 복사
COPY . .

# Whisper 모델 다운로드 및 초기화
RUN python3 -c "from faster_whisper import WhisperModel; model = WhisperModel('large-v3')"

# 실행 명령
CMD ["python", "runpod_main.py"]