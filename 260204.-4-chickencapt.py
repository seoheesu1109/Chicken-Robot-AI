import cv2
import numpy as np
import pyrealsense2 as rs
import os
import time

# [1] 저장 설정
save_path = '260204-realchicken_data'
if not os.path.exists(save_path):
    os.makedirs(save_path)

# [2] 리얼센서 파이프라인 설정
pipeline = rs.pipeline()
config = rs.config()

# 컬러 스트림 설정 (학습용이므로 고해상도 1280x720 추천)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
# 필요시 Depth도 함께 설정 가능
# config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)

# 스트리밍 시작
profile = pipeline.start(config)

count = 0
print("--- RealSense 캡처 시작 ---")
print("'s' 키를 누르면 자동 저장을 시작/중지합니다.")
print("'q' 키를 누르면 종료합니다.")

is_capturing = False # 처음엔 대기 상태

try:
    while True:
        # 프레임 대기
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        
        if not color_frame:
            continue

        # 이미지 데이터를 numpy 배열로 변환
        frame = np.asanyarray(color_frame.get_data())

        # 화면 표시
        cv2.imshow('RealSense Capturing...', frame)
        
        key = cv2.waitKey(1)
        
        # 's' 키로 캡처 토글 (시작/정지)
        if key == ord('s'):
            is_capturing = not is_capturing
            state = "시작" if is_capturing else "중지"
            print(f"자동 캡처 {state}...")

        # 'q' 키로 종료
        elif key == ord('q'):
            break

        # 캡처 모드일 때 0.5초마다 저장
        if is_capturing:
            file_name = f"{save_path}/drumstick_{count}.jpg"
            cv2.imwrite(file_name, frame)
            print(f"저장됨: {file_name} (Total: {count})")
            count += 1
            time.sleep(0.5) # 0.5초 대기

finally:
    # 스트리밍 중지
    pipeline.stop()
    cv2.destroyAllWindows()