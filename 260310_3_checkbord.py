import cv2
import numpy as np

# 저장된 데이터 불러오기
data = np.load('calibration_data.npz')
mtx, dist = data['mtx'], data['dist']

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

while True:
    ret, frame = cap.read()
    if not ret: break
    
    h, w = frame.shape[:2]
    # 새로운 카메라 매트릭스 계산 (보정 시 잘려나가는 부분 제어)
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
    
    # 왜곡 보정 (Undistort)
    dst = cv2.undistort(frame, mtx, dist, None, newcameramtx)
    
    # 보정 전/후 비교 표시
    cv2.imshow('Original (Before)', frame)
    cv2.imshow('Undistorted (After)', dst)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()