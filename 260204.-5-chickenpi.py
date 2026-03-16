import cv2
import numpy as np
from ultralytics import YOLO

# 1. YOLO 모델 로드 (학습된 best.pt 경로 확인)
model_path = r"C:\runs\detect\train12\weights\best.pt"
model = YOLO(model_path)

# 2. 일반 웹캠 설정
# 0번은 기본 내장 카메라, 외장 웹캠 사용 시 1번 또는 2번으로 시도하세요.
cap = cv2.VideoCapture(0)

# 해상도 설정 (웹캠 지원 사양에 맞게 조절)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("🚀 일반 웹캠을 활용한 YOLO 치킨 감지를 시작합니다! (종료: 'q')")

try:
    while True:
        # 웹캠 프레임 읽기
        ret, frame = cap.read()
        if not ret:
            print("❌ 카메라 프레임을 읽을 수 없습니다.")
            break

        # 3. YOLO 모델로 감지 (신뢰도 조절 가능)
        results = model.predict(
            frame, 
            conf=0.5,          # 신뢰도 임계값
            iou=0.4,           # NMS(중복 제거) 임계값
            agnostic_nms=True, # 클래스 간 중복 제거
            verbose=False      # 터미널 로그 간소화
        )

        # 결과 그리기 (원본 프레임 복사)
        annotated_frame = frame.copy()

        for r in results:
            # 화면에 박스 그리기 (YOLO 기본 시각화)
            annotated_frame = r.plot()
            
            for box in r.boxes:
                # 박스 중심점 계산
                xyxy = box.xyxy[0].cpu().numpy()
                x_center = int((xyxy[0] + xyxy[2]) / 2)
                y_center = int((xyxy[1] + xyxy[3]) / 2)
                
                # 클래스 이름과 신뢰도 가져오기
                cls = int(box.cls[0])
                label = model.names[cls]
                conf = box.conf[0]

                # [출력] 일반 웹캠은 거리(Z)가 없으므로 픽셀 좌표만 출력
                print(f"🎯 [{label}] 감지! 중심 픽셀: ({x_center}, {y_center}) | 신뢰도: {conf:.2f}")

                # 중심점에 점 찍기 (시각화 강조)
                cv2.circle(annotated_frame, (x_center, y_center), 5, (0, 0, 255), -1)

        # 4. 결과 화면 출력
        cv2.imshow("Webcam YOLO Chicken Detection", annotated_frame)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # 자원 해제
    cap.release()
    cv2.destroyAllWindows()