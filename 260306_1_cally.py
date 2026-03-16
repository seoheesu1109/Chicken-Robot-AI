import cv2
import cv2.aruco as aruco
import numpy as np

# 카메라 설정 (이전에 사용하던 인덱스 1번)
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# 아르코 마커 설정
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
params = aruco.DetectorParameters()

# 내장된 카메라 행렬 (기존 사용값)
camera_matrix = np.array([[900.0, 0, 640.0], [0, 900.0, 360.0], [0, 0, 1]], dtype=float)
dist_coeffs = np.zeros((5, 1))

print("=== 캘리브레이션 데이터 수집 모드 ===")
print("마커를 감지하면 화면에 X, Y, Z 좌표가 나타납니다.")
print("'q'를 누르면 종료합니다.")

while True:
    ret, frame = cap.read()
    if not ret: break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=params)

    if ids is not None:
        # 마커의 자세 추정 (마커 크기 100mm 기준)
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, 100, camera_matrix, dist_coeffs)
        
        for i in range(len(ids)):
            # tvec[0][0]이 우리가 필요한 [X, Y, Z] 카메라 좌표입니다.
            curr_tvec = tvecs[i][0]
            
            # 화면에 마커 테두리 및 좌표 표시
            aruco.drawDetectedMarkers(frame, corners, ids)
            cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvecs[i], tvecs[i], 50)
            
            text = f"CAM DATA (X: {curr_tvec[0]:.2f}, Y: {curr_tvec[1]:.2f}, Z: {curr_tvec[2]:.2f})"
            cv2.putText(frame, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # 터미널에도 출력 (복사하기 편하게)
            print(f"ID: {ids[i][0]} -> [{curr_tvec[0]:.2f}, {curr_tvec[1]:.2f}, {curr_tvec[2]:.2f}]")

    cv2.imshow("Calibration Data Collector", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()