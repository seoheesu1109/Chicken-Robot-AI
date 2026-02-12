import numpy as np
import cv2
import pyrealsense2 as rs

# [설정값] - 이전과 동일
CAMERA_Z_OFFSET = 90.0
X_BIAS = -82.13
Y_BIAS = -27.10
Z_BIAS = 344.15

# [하늘색 HSV 범위] - 실제 환경에 따라 미세 조정 필요
# OpenCV HSV: H(색상), S(채도), V(명도)
lower_cyan = np.array([85, 100, 100])
upper_cyan = np.array([105, 255, 255])

def get_final_calibrated_target(p_cam, robot_now):
    cam_x, cam_y, cam_z = p_cam
    curr_x, curr_y, curr_z = robot_now
    rel_x, rel_y = -cam_x, -cam_y
    rel_z = -(cam_z - CAMERA_Z_OFFSET)
    return curr_x + rel_x + X_BIAS, curr_y + rel_y + Y_BIAS, curr_z + rel_z + Z_BIAS

# 리얼센스 초기화
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
profile = pipeline.start(config)
align = rs.align(rs.stream.color)
intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

print("🔵 하늘색 공 추적 시스템 가동")
print("🔘 'b' 키: 공 중심의 로봇 좌표 계산")

try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned = align.process(frames)
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()
        if not color_frame or not depth_frame: continue

        img = np.asanyarray(color_frame.get_data())
        # 노이즈 제거를 위한 블러 처리
        blurred = cv2.GaussianBlur(img, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        # 하늘색 마스크 생성
        mask = cv2.inRange(hsv, lower_cyan, upper_cyan)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        # 윤곽선 찾기
        cnts, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        center = None

        if len(cnts) > 0:
            # 가장 큰 윤곽선(공) 찾기
            c = max(cnts, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            
            if M["m00"] > 0:
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                
                # 원과 중심점 그리기
                if radius > 10:
                    cv2.circle(img, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                    cv2.circle(img, center, 5, (0, 0, 255), -1)

                    # 'b' 키 입력 시 좌표 계산
                    if cv2.waitKey(1) == ord('b'):
                        u, v = center
                        depth = depth_frame.get_distance(u, v) * 1000
                        if depth > 0:
                            p_cam = rs.rs2_deproject_pixel_to_point(intr, [u, v], depth)
                            # 현재 로봇 좌표 (마지막 주신 값 기준)
                            current_robot_pos = [800.49, 912.48, 404.41]
                            tx, ty, tz = get_final_calibrated_target(p_cam, current_robot_pos)
                            
                            print(f"\n🔵 공 중심 로봇 좌표 -> X:{tx:.2f}, Y:{ty:.2f}, Z:{tz:.2f}")

        cv2.imshow("Blue Ball Tracking", img)
        if cv2.waitKey(1) == ord('q'): break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()