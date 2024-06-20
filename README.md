# youtube_summary_chatbot

## 아이디어 기록

```
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
