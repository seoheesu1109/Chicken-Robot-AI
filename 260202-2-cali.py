import cv2
import cv2.aruco as aruco
import numpy as np
import pyrealsense2 as rs

# ==========================================
# 1. 수집된 데이터 입력 (Camera M -> Robot R)
# ==========================================
# M: 카메라 기준 [mx, my, mz], R: 로봇 기준 [rx, ry, rz]
M_pts = np.array([
    [-61.48, -146.00, 1653.00],
    [87.49, -10.75, 1558.11],
    [-408.00, 122.00, 1439.00],
    [-222.00, -307.00, 1070.00]
], dtype=np.float32)

R_pts = np.array([
    [860.92, 1073.92, 161.83],
    [1102.16, 1067.30, 154.97],
    [837.28, 630.72, 154.97],
    [837.28, 868.99, 679.61]
], dtype=np.float32)

# 3D 변환 행렬(Affine Matrix) 계산
# OpenCV 버전에 따른 반환값 개수 차이 해결
res = cv2.estimateAffine3D(M_pts, R_pts)
if len(res) == 3:
    retval, matrix, inliers = res
else:
    matrix, inliers = res

if matrix is None:
    print("❌ 캘리브레이션 행렬 계산 실패! 데이터 세트를 확인하세요.")
    exit()

def camera_to_robot(mx, my, mz):
    """카메라의 mx, my, mz를 로봇의 rx, ry, rz로 변환"""
    src_pt = np.array([[[mx, my, mz]]], dtype=np.float32)
    dst_pt = cv2.transform(src_pt, matrix)
    return dst_pt[0][0] # [rx, ry, rz] 반환

# ==========================================
# 2. 리얼센스 및 아루코 마커 설정
# ==========================================
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
profile = pipeline.start(config)

# 카메라 내인성 파라미터(렌즈 특성) 가져오기
intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
camera_matrix = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]])
dist_coeffs = np.array(intr.coeffs)

# 마커 설정 (사이즈 100mm)
marker_length = 100 
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()

print("✅ 캘리브레이션 검증 모드 가동 (종료: q)")
print("마커를 움직이며 화면 상단의 Robot X, Y, Z 값이 로봇 좌표와 맞는지 확인하세요.")

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame: continue
        
        img = np.asanyarray(color_frame.get_data())

        # 마커 감지
        corners, ids, _ = aruco.detectMarkers(img, aruco_dict, parameters=parameters)

        if ids is not None:
            # 마커의 3D 포즈(위치 및 회전) 추정
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, marker_length, camera_matrix, dist_coeffs)
            
            for i in range(len(ids)):
                # 카메라 기준 마커 좌표 (mx, my, mz)
                mx, my, mz = tvecs[i][0]
                
                # 로봇 기준 좌표로 변환 (rx, ry, rz)
                rx, ry, rz = camera_to_robot(mx, my, mz)
                
                # 화면 시각화
                cv2.drawFrameAxes(img, camera_matrix, dist_coeffs, rvecs[i], tvecs[i], 50)
                
                # 좌표 텍스트 출력
                label = f"Robot Pos -> X:{rx:.1f} Y:{ry:.1f} Z:{rz:.1f}"
                cv2.putText(img, label, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
                # 터미널 실시간 출력
                print(f"🤖 [Live] Robot Coordinates: X={rx:.2f}, Y={ry:.2f}, Z={rz:.2f}")

        cv2.imshow("Hand-Eye Calibration Verify", img)
        
        if cv2.waitKey(1) == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()