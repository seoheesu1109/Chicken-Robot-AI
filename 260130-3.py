import cv2                      # OpenCV 메인 라이브러리
import numpy as np               # 행렬 계산용

# 카메라 내부 파라미터 (캘리브레이션 결과)
camera_matrix = np.array([
    [1000, 0, 640],
    [0, 1000, 360],
    [0, 0, 1]
], dtype=np.float32)

# 렌즈 왜곡 계수 (없으면 0으로)
dist_coeffs = np.zeros((5, 1))

# ArUco 딕셔너리 선택 (가장 많이 쓰는 4x4 50개)
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

# 마커 검출 파라미터 생성
aruco_params = cv2.aruco.DetectorParameters()

cap = cv2.VideoCapture(0)   # 0번 웹캠 열기

marker_length = 0.05  # 마커 한 변 길이 (단위: 미터) → 5cm

while True:
    ret, frame = cap.read()     # 카메라 프레임 읽기
    if not ret:
        break

    # BGR → Grayscale 변환 (ArUco는 흑백에서 더 잘 인식됨)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    corners, ids, rejected = cv2.aruco.detectMarkers(
        gray,
        aruco_dict,
        parameters=aruco_params
    )
    if ids is not None:
        # 검출된 마커 프레임에 표시
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners,          # 마커 코너 좌표
            marker_length,    # 마커 실제 크기
            camera_matrix,    # 카메라 내부 파라미터
            dist_coeffs       # 왜곡 계수
        )
        for i in range(len(ids)):
            rvec = rvecs[i]
            tvec = tvecs[i]

            # 위치 좌표 추출
            x, y, z = tvec[0]

            # 콘솔 출력
            print(f"ID {ids[i][0]} | X:{x:.3f} Y:{y:.3f} Z:{z:.3f}")

            # 마커 좌표축 시각화
            cv2.drawFrameAxes(
                frame,
                camera_matrix,
                dist_coeffs,
                rvec,
                tvec,
                0.03   # 축 길이 (미터)
            )
    cv2.imshow("ArUco Pose", frame)

    # ESC 키 누르면 종료
    if cv2.waitKey(1) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()
