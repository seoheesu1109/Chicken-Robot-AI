import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO

# 1. YOLO 모델 로드
model_path = r"C:\runs\detect\train15\weights\best.pt"
model = YOLO(model_path)

# 2. 리얼센서 파이프라인 및 설정
pipeline = rs.pipeline()
config = rs.config()

# 컬러 스트림 설정 (YOLO 학습 해상도에 맞춰 조절 가능)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30) # 거리 확인용 뎁스

# 스트리밍 시작
pipeline.start(config)
align = rs.align(rs.stream.color) # 뎁스와 컬러 좌표 정렬용

print("🚀 RealSense를 활용한 치킨 감지를 시작합니다! (종료: 'q')")

try:
    while True:
        # 프레임 세트 대기
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        
        if not color_frame or not depth_frame:
            continue

        # 이미지 데이터를 numpy 배열로 변환 (YOLO 입력용)
        frame = np.asanyarray(color_frame.get_data())

        # 3. YOLO 모델로 감지 (신뢰도 0.5 이상)
        results = model.predict(
        frame, 
        conf=0.15,    # 신뢰도를 조금 낮춰서 더 많이 찾게 함
        iou=0.45,     # 박스끼리 겹치는 것을 얼마나 허용할지 (0.45~0.6 추천)
        agnostic_nms=True, # 클래스 간 중복 제거 활성화
        verbose=False
    )

        annotated_frame = frame.copy() # 원본 복사하여 결과 그리기

        for r in results:
            # 화면에 박스 그리기 (YOLO 기본 제공 기능)
            annotated_frame = r.plot()
            
            for box in r.boxes:
                # 박스 중심점 계산
                xyxy = box.xyxy[0].cpu().numpy()
                x_center = int((xyxy[0] + xyxy[2]) / 2)
                y_center = int((xyxy[1] + xyxy[3]) / 2)
                
                # [RealSense 추가 기능] 중심점의 실제 거리(Z) 가져오기
                dist = depth_frame.get_distance(x_center, y_center) * 1000 # mm 단위
                
                print(f"🎯 감지! 중심: ({x_center}, {y_center}) | 거리: {dist:.1f}mm")

        # 4. 결과 화면 출력
        cv2.imshow("RealSense YOLO Chicken Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # 파이프라인 종료
    pipeline.stop()
    cv2.destroyAllWindows()