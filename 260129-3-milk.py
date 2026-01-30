import cv2
from ultralytics import YOLO

# 1. 방금 만든 따끈따끈한 모델 불러오기 (경로 내 공백 주의!)
# r을 붙여서 경로 내 공백과 한글을 있는 그대로 인식하게 합니다.
model_path = r"C:\runs\detect\train6\weights\best.pt"
model = YOLO(model_path)

# 2. 카메라 켜기
cap = cv2.VideoCapture(0)

print("카메라를 켭니다. 물체를 비춰보세요! (종료하려면 'q'를 누르세요)")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # 3. 모델로 물체 감지 (신뢰도 0.5 이상만)
    results = model.predict(frame, conf=0.5)

    for r in results:
        # 화면에 박스 그리기
        annotated_frame = r.plot()
        
        # 감지된 물체가 있다면 좌표 출력
        for box in r.boxes:
            # 중심점 계산
            xyxy = box.xyxy[0]
            x_center = (xyxy[0] + xyxy[2]) / 2
            y_center = (xyxy[1] + xyxy[3]) / 2
            
            print(f"🎯 감지 성공! 중심좌표: X={x_center:.1f}, Y={y_center:.1f}")

    # 4. 화면 보여주기
    cv2.imshow("Milk Detection Test", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()