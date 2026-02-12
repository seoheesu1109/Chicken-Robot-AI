import numpy as np
import cv2
import pyrealsense2 as rs

# ==========================================
# 1. 수집된 데이터 입력 (카메라 -> 로봇 매칭)
# ==========================================
# 카메라에서 본 상대 좌표 [x, y, z]
C_pts = np.array([
    [-60.99, -22.08, 751.00],   # 1번 점
    [316.67, 13.45, 631.00],    # 2번 점
    [-413.18, -43.74, 623.00],  # 3번 점
    [-12.00, -276.04, 694.00],  # 4번 점
    [-74.52, 253.81, 714.00]    # 5번 점
], dtype=np.float32)

# 로봇 펜던트에 찍힌 실제 좌표 [X, Y, Z]
# (1번 점 Z값이 누락되어 평균치인 40 정도로 가정했습니다. 실제 값으로 수정 가능합니다.)
R_pts = np.array([
    [775.38, 893.70, 40.00],    # 1번 점
    [410.45, 914.97, 178.26],   # 2번 점
    [1142.18, 890.76, 174.17],  # 3번 점
    [741.48, 641.75, 97.46],    # 4번 점
    [789.80, 1178.78, 95.67]    # 5번 점
], dtype=np.float32)

# [핵심] 3D 아핀 변환 행렬 계산
# 이 M 행렬이 카메라 세상을 로봇 세상으로 바꾸는 '지도' 역할을 합니다.
retval, M, inliers = cv2.estimateAffine3D(C_pts, R_pts)

def get_robot_world_coords(p_cam):
    """카메라 좌표 [x, y, z]를 넣으면 로봇 절대 좌표 [X, Y, Z]를 반환"""
    point = np.append(p_cam, 1.0)
    target = M.dot(point)
    return target

# ==========================================
# 2. 실시간 감지 및 좌표 변환 루프
# ==========================================
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
profile = pipeline.start(config)
align = rs.align(rs.stream.color)
intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

# 하늘색 공 범위
lower_cyan = np.array([85, 100, 100])
upper_cyan = np.array([105, 255, 255])

print("🎯 캘리브레이션 행렬 적용 완료. 공을 감지하면 'g'를 누르세요.")

try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned = align.process(frames)
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()
        if not color_frame or not depth_frame: continue

        img = np.asanyarray(color_frame.get_data())
        hsv = cv2.cvtColor(cv2.GaussianBlur(img, (11, 11), 0), cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_cyan, upper_cyan)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(cnts) > 0:
            c = max(cnts, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            if radius > 15:
                u, v = int(x), int(y)
                cv2.circle(img, (u, v), int(radius), (0, 255, 0), 2)

                if cv2.waitKey(1) == ord('g'):
                    depth = depth_frame.get_distance(u, v) * 1000
                    p_cam = rs.rs2_deproject_pixel_to_point(intr, [u, v], depth)
                    
                    # 행렬 연산으로 로봇 좌표 계산
                    target_X, target_Y, target_Z = get_robot_world_coords(p_cam)
                    
                    print(f"\n🚀 [이동 명령 생성]")
                    print(f"변환 결과: X:{target_X:.2f}, Y:{target_Y:.2f}, Z:{target_Z:.2f}")
                    print(f"👉 이 좌표로 로봇을 이동시키면 됩니다!")

        cv2.imshow("Final Robot Vision", img)
        if cv2.waitKey(1) == ord('q'): break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()