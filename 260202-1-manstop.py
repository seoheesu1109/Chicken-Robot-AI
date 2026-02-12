import socket, time, threading
import cv2, numpy as np
import pyrealsense2 as rs
from ultralytics import YOLOWorld

class UltimateRobotSafety:
    def __init__(self):
        self.is_emergency = False
        self.robot_ready_event = threading.Event()
        self.robot_ready_event.set() # 처음 시작 시 이동 가능하도록 설정
        self.current_idx = 0
        self.conn = None
        
        print("🚀 모델 로딩 중...")
        self.model = YOLOWorld('yolov8s-worldv2.pt') 
        self.model.set_classes(["person"])
        
        # 소켓 서버 설정 (포트 30002)
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind(("0.0.0.0", 30002))
        self.server.listen(1)

    def robot_feedback_receiver(self):
        """로봇으로부터 도착 완료(DONE) 신호를 받는 함수"""
        while True:
            if self.conn:
                try:
                    raw_data = self.conn.recv(1024)
                    if not raw_data: break
                    data = raw_data.decode().strip()
                    if "DONE" in data:
                        print(f"✅ [로봇 응답] {self.current_idx}번 지점 도착 완료 확인")
                        self.robot_ready_event.set() 
                except:
                    break

    def sequence_worker(self):
        """좌표를 하나씩 순서대로 전송하는 함수"""
        TARGET_POINTS = [[1000.0, 100.0], [1200.0, -200.0], [800.0, 50.0]]
        while True:
            # 안전 모드가 아니고, 로봇이 이전 동작을 마쳤을 때만 실행
            if not self.is_emergency and self.robot_ready_event.is_set():
                self.robot_ready_event.clear() 
                
                p = TARGET_POINTS[self.current_idx]
                msg = f"MOVE,{p[0]},{p[1]},700.0,180.0,-180.0,90.0\n"
                
                try:
                    self.conn.sendall(msg.encode())
                    print(f"🚀 [명령 전송] {self.current_idx + 1}번 지점으로 출발: {p}")
                except Exception as e:
                    print(f"❌ 전송 에러: {e}")
                
                self.current_idx = (self.current_idx + 1) % len(TARGET_POINTS)
            time.sleep(0.1)

    def camera_worker(self):
        pipeline = rs.pipeline()
        config = rs.config()
        # 여기서 bgr8로 설정했으므로, 아래에서 cvtColor가 필요 없을 확률이 높습니다.
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        pipeline.start(config)
        
        try:
            while True:
                frames = pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                if not color_frame: continue
                
                img = np.asanyarray(color_frame.get_data())
                
                # 🚨 여기서 색상이 반전된다면 아래 줄을 지우거나 주석처리(#) 하세요!
                # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) 

                results = self.model.predict(img, conf=0.7, verbose=False)
                
                if len(results[0].boxes) > 0:
                    if not self.is_emergency:
                        self.is_emergency = True
                        if self.conn: self.conn.sendall(b"STOP\n")
                        print("🚨 [위험] 사람 감지 - 즉시 정지 신호 전송!")
                else:
                    if self.is_emergency:
                        print("✅ [안전] 사람이 나갔습니다. 작업을 재개합니다.")
                        self.is_emergency = False
                        self.robot_ready_event.set() # 멈췄던 동작 다시 시작

                cv2.imshow("Safety Monitor", results[0].plot())
                if cv2.waitKey(1) == ord('q'): break
        finally:
            pipeline.stop()
            cv2.destroyAllWindows()

    # 🚨 이 start 함수가 클래스 안(안쪽으로 들여쓰기)에 있어야 에러가 안 납니다!
    def start(self):
        print("📡 로봇 연결 대기 중 (포트 30002)...")
        self.conn, addr = self.server.accept()
        print(f"🤝 로봇 연결 성공: {addr}")
        
        # 3개의 스레드 실행
        threading.Thread(target=self.robot_feedback_receiver, daemon=True).start()
        threading.Thread(target=self.camera_worker, daemon=True).start()
        threading.Thread(target=self.sequence_worker, daemon=True).start()
        
        while True:
            time.sleep(1)

# 메인 실행부
if __name__ == "__main__":
    app = UltimateRobotSafety()
    app.start()