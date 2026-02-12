import numpy as np
import cv2
import pyrealsense2 as rs

# [1] 6점 피라미드 기반 행렬 (고정 데이터)
C_pts = np.array([[560,367,797], [566,112,781], [136,436,795], [821,429,803], [552,620,802], [427,401,288]], dtype=np.float32)
R_pts = np.array([[728.9,947.73,742.91], [728.9,1167.84,742.91], [365.18,947.73,742.91], [961.49,947.73,742.91], [728.9,724.8,742.9], [728.9,947.7,240.9]], dtype=np.float32)

res = cv2.estimateAffine3D(C_pts, R_pts)
HAND_EYE_MATRIX = res[1] if len(res) == 3 else res[0]

# [2] 실측 기반 정밀 보정값 (오늘의 핵심)
X_OFFSET = 559.32
Y_OFFSET = -533.21
Z_OFFSET = -10.46

# [3] 현재 안전 위치 로봇 좌표 (고정)
SAFE_ROBOT_POS = [800.49, 912.48, 401.41] 

def get_calibrated_coord(p_cam, robot_pos):
    src_pt = np.array([[[p_cam[0], p_cam[1], p_cam[2]]]], dtype=np.float32)
    dst_pt = cv2.transform(src_pt, HAND_EYE_MATRIX)
    
    # 6점 행렬 통과 후 나온 생(Raw) 좌표
    raw_x, raw_y, raw_z = dst_pt[0][0]
    
    # 로봇의 상대적 이동량 계산
    diff_x = robot_pos[0] - R_pts[0][0]
    diff_y = robot_pos[1] - R_pts[0][1]
    diff_z = robot_pos[2] - R_pts[0][2]
    
    # 최종 결과 = (행렬 결과 + 이동량) + 실측 오차 보정
    final_x = raw_x + diff_x + X_OFFSET
    final_y = raw_y + diff_y + Y_OFFSET
    final_z = raw_z + diff_z + Z_OFFSET
    
    return final_x, final_y, final_z

# --- 리얼센스 및 ArUco 설정 ---
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
profile = pipeline.start(config)
align = rs.align(rs.stream.color)
intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100))

print("🎯 보정 완료! 이제 화면의 좌표가 실제 로봇 좌표와 동기화됩니다.")

try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned = align.process(frames)
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()
        if not color_frame or not depth_frame: continue

        img = np.asanyarray(color_frame.get_data())
        corners, ids, _ = detector.detectMarkers(img)

        if ids is not None:
            u, v = int(corners[0][0][:, 0].mean()), int(corners[0][0][:, 1].mean())
            depth = depth_frame.get_distance(u, v) * 1000
            
            if depth > 0:
                p_cam = rs.rs2_deproject_pixel_to_point(intr, [u, v], depth)
                rx, ry, rz = get_calibrated_coord(p_cam, SAFE_ROBOT_POS)
                
                # 이 좌표를 로봇에게 전송하면 정확히 마커 위로 갑니다.
                print(f"✅ 동기화된 좌표 -> X:{rx:.2f}, Y:{ry:.2f}, Z:{rz:.2f}")

        cv2.imshow("Final Calibration Sync", img)
        if cv2.waitKey(1) == ord('q'): break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()