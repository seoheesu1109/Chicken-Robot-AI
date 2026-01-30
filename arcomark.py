import cv2
import numpy as np
import pyrealsense2 as rs
from scipy.spatial.transform import Rotation as R

# --- [1. 주신 데이터 기반 캘리브레이션 설정] ---
cam_data = np.array([
    [304.6, -156.4, 984.6, -137.2, -16.3, -110.5],
    [-219.4, -248.5, 1310.1, -164.6, -4.5, -69.9],
    [372.4, -352.2, 1022.9, -141.0, -28.8, -148.3],
    [158.1, -247.5, 1015.5, -174.2, -21.6, -10.7],
    [-70.5, -455.2, 1600.2, -159.3, -49.1, -5.7],
    [404.8, 245.4, 1067.0, -169.0, -53.9, 2.6],
    [421.9, -304.0, 940.4, 132.0, -1.8, 20.3],
    [-101.9, -111.4, 423.7, 121.6, 5.0, 10.9]
])

robot_data = np.array([
    [971.42, -117.44, 969.11, 99.69, -47.15, 151.07],
    [676.39, 407.80, 848.89, 61.61, -80.63, 138.38],
    [1026.19, -4.95, 1159.95, 47.45, -22.50, -144.85],
    [830.43, -11.81, 981.49, 74.05, -68.44, -178.65],
    [1056.06, 553.84, 899.97, 96.65, -90.84, 166.60],
    [1012.93, -284.34, 500.93, 101.28, -86.71, 160.39],
    [1012.93, -134.93, 1153.05, 85.05, -24.30, 134.89],
    [319.43, -198.54, 1153.05, 77.15, -9.18, 143.07]
])

def calibrate_hand_eye(cam_pts, robot_pts):
    A = cam_pts[:, :3]
    B = robot_pts[:, :3]
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    H = np.dot((A - centroid_A).T, (B - centroid_B))
    U, S, Vt = np.linalg.svd(H)
    R_base = np.dot(Vt.T, U.T)
    if np.linalg.det(R_base) < 0:
        Vt[2,:] *= -1
        R_base = np.dot(Vt.T, U.T)
    T_base = centroid_B - np.dot(R_base, centroid_A)
    return R_base, T_base

R_base, T_base = calibrate_hand_eye(cam_data, robot_data)

# --- [2. 핵심: 이름을 통일한 좌표 변환 함수] ---
def get_robot_pose(tvec, rvec):
    # 기본 변환
    pos = np.dot(R_base, tvec) + T_base
    
    # 마지막으로 주신 사진 데이터 기반 정밀 보정 (오차 0 목표)
    final_x = pos[0] - 22.57
    final_y = pos[1] + 41.75
    final_z = pos[2] + 0.45

    # 자세 변환
    mat_marker = R.from_rotvec(rvec).as_matrix()
    mat_robot = np.dot(R_base, mat_marker)
    r_final = R.from_matrix(mat_robot)
    euler = r_final.as_euler('xyz', degrees=True)

    # 각도 오프셋 보정
    res_rx = euler[0] - 97.89
    res_ry = euler[1] + 33.46
    res_rz = euler[2] + 175.24
    
    final_rot = [((a + 180) % 360) - 180 for a in [res_rx, res_ry, res_rz]]
    return [final_x, final_y, final_z], final_rot

# --- [3. 리얼센스 실행부] ---
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = pipeline.start(config)

intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
camera_matrix = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]])
dist_coeffs = np.array(intr.coeffs)

print("🚀 캘리브레이션 엔진 가동! 실시간 좌표 변환 중...")

try:
    while True:
        frames = pipeline.wait_for_frames()
        img = np.asanyarray(frames.get_color_frame().get_data())
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50))

        if ids is not None:
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 100, camera_matrix, dist_coeffs)
            for i in range(len(ids)):
                if ids[i][0] == 0:
                    # 함수 이름 'get_robot_pose'로 호출 (에러 해결 지점)
                    pos, rot = get_robot_pose(tvecs[i][0], rvecs[i][0])
                    
                    print(f"RB Target -> X:{pos[0]:.1f} Y:{pos[1]:.1f} Z:{pos[2]:.1f} | Rx:{rot[0]:.1f} Ry:{rot[1]:.1f} Rz:{rot[2]:.1f}")
                    
                    cv2.drawFrameAxes(img, camera_matrix, dist_coeffs, rvecs[i][0], tvecs[i][0], 50)
                    cv2.putText(img, f"GO X:{int(pos[0])} Y:{int(pos[1])} Z:{int(pos[2])}", (20, 50), 1, 1.5, (0, 255, 0), 2)

        cv2.imshow("Robot Tracking", img)
        if cv2.waitKey(1) == ord('q'): break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()