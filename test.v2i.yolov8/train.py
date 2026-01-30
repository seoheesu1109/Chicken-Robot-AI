from ultralytics import YOLO
import os

if __name__ == "__main__":
    # 1. 모델 불러오기 (가장 가벼운 n 모델)
    model = YOLO("yolov8n.pt") 

    # 2. data.yaml 파일의 절대 경로 (예: r"C:\Users\Desktop\project\data.yaml")
    # 아래 경로를 본인의 실제 경로로 꼭 수정하세요!
    yaml_path = r"C:\Users\gmltn\OneDrive\바탕 화면\doosanpy\test.v2i.yolov8\data.yaml"

    # 3. 학습 시작
    model.train(
        data=yaml_path,
        epochs=30,      # 30번 반복
        imgsz=640,      # 이미지 크기
        device="cpu",   # 외장 그래픽카드가 없다면 cpu
        project="my_robot",
        name="drumstick_v1"
    )