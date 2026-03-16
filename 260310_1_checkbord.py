import cv2
import os
import time

# --- 설정 ---
CHECKERBOARD = (9, 6)  # 체커보드 가로/세로 내부 교차점 개수 (수정 필요할 수 있음)
SAVE_PATH = 'calibration_images'
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW) # 카메라 안나오면 0으로 변경
count = 0
last_save_time = time.time()

print(f"자동 캡처 시작! 체커보드를 카메라에 비추세요.")
print(f"인식되면 2초마다 자동 저장됩니다. (목표: 20장) / 종료: 'Q'")

while count < 20:
    ret, frame = cap.read()
    if not ret: break
    
    display_frame = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 체커보드 찾기
    ret_found, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
    
    if ret_found:
        # 화면에 인식 표시 (초록색 점)
        cv2.drawChessboardCorners(display_frame, CHECKERBOARD, corners, ret_found)
        cv2.putText(display_frame, "PATTERN DETECTED!", (30, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 2초 간격으로 자동 저장
        current_time = time.time()
        if current_time - last_save_time > 2.0:
            filename = f"{SAVE_PATH}/calib_{count}.jpg"
            cv2.imwrite(filename, frame)
            count += 1
            last_save_time = current_time
            print(f"[{count}/20] 사진 저장 완료: {filename}")
            
    cv2.putText(display_frame, f"Saved: {count}/20", (30, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    cv2.imshow("Auto Capture Mode", display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print("캡처가 완료되었습니다.")
cap.release()
cv2.destroyAllWindows()