import cv2
import numpy as np

def nothing(x):
    pass

# 제어창 생성 (감도 조절용)
cv2.namedWindow("Settings")
cv2.createTrackbar("Threshold1", "Settings", 50, 255, nothing)
cv2.createTrackbar("Threshold2", "Settings", 150, 255, nothing)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.resize(frame, (640, 480))
    # 1. 그레이스케일 변환 (색상 정보를 버리고 명암만 남김)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 2. 노이즈 제거 (외곽선을 깔끔하게 따기 위해 필수)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 3. Canny 에지 검출 (슬라이더로 감도 조절 가능)
    t1 = cv2.getTrackbarPos("Threshold1", "Settings")
    t2 = cv2.getTrackbarPos("Threshold2", "Settings")
    edged = cv2.Canny(blurred, t1, t2)

    # 4. 에지 연결 및 팽창 (끊어진 선을 붙여줌)
    kernel = np.ones((3,3), np.uint8)
    dilated = cv2.dilate(edged, kernel, iterations=1)

    # 5. 외곽선 찾기
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1500: # 너무 작은 먼지는 무시
            # 초록색 외곽선 그리기
            cv2.drawContours(frame, [cnt], -1, (0, 255, 0), 2)

            # 장축 및 각도 계산
            rect = cv2.minAreaRect(cnt)
            (cx, cy), (w, h), angle = rect
            
            # 사각형 그리기
            box = cv2.boxPoints(rect)
            box = box.astype(int)
            cv2.drawContours(frame, [box], 0, (255, 0, 0), 2)

            # 각도 보정 및 표시
            if w < h: angle += 90
            cv2.putText(frame, f"Angle: {int(angle)}deg", (int(cx), int(cy)-20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow("Every Object Detection", frame)
    cv2.imshow("Edged (What AI sees)", dilated) # AI가 선을 어떻게 따고 있는지 확인용

    if cv2.waitKey(1) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()