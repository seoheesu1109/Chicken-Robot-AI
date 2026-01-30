# 🍗 AI Vision 기반 치킨 조리 로봇 제어 시스템

YOLO-World 모델을 활용하여 별도의 학습 데이터 없이도 다양한 메뉴를 인식하고, 
두산 협동 로봇(Doosan Robot)을 제어하여 자동 배출 공정을 수행하는 프로젝트입니다.

## 📺 프로젝트 데모 영상
(<video cont

https://github.com/user-attachments/assets/fc8a0eca-2450-4391-9a27-aaa532202326

rols src="chicken1.mp4" title="Title"></video>)

## 🚀 주요 기능
- **Zero-shot Object Detection**: YOLO-World를 사용하여 "Chicken", "Fries" 등 텍스트 입력만으로 사물 인식
- **Real-time Robot Control**: 웹캠 좌표를 로봇 작업 좌표계(TCP)로 변환하여 실시간 이동 제어
- **Safety Logic**: 특정 위험 구역(튀김기 벽면 등) 감지 시 자동 회피 및 대기 로직 적용

## 🛠 Tech Stack
- **Language**: Python 3.13
- **AI Model**: YOLO-World (v8s)
- **Library**: OpenCV, Ultralytics, Socket (TCP/IP)
- **Hardware**: Doosan Robotics

## ⚙️ 실행 방법
1. 필요 라이브러리 설치
   ```bash
   pip install ultralytics opencv-python
