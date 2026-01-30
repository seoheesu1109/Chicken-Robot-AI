import cv2
import time
import os

# 저장할 폴더 생성
save_path = 'realchicken_data'
if not os.path.exists(save_path):
    os.makedirs(save_path)

cap = cv2.VideoCapture(0) # 0번 웹캠 연결
count = 0

print("--- 캡처를 시작합니다. 'q'를 누르면 종료됩니다. ---")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 화면에 현재 모습 표시
        cv2.imshow('Capturing...', frame)

        # 0.5초마다 이미지 저장 (숫자는 조절 가능)
        file_name = f"{save_path}/drumstick_{count}.jpg"
        cv2.imwrite(file_name, frame)
        print(f"저장됨: {file_name}")
        
        count += 1
        time.sleep(0.5) # 0.5초 대기

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()