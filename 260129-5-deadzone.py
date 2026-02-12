import socket
import threading
import time
import tkinter as tk
from tkinter import messagebox
import numpy as np
import cv2
import pyrealsense2 as rs
from ultralytics import YOLO

# [설정 데이터]
PORT = 30002
Z_CORRECTION = 75.0 
HOME_POS = [647.92, -119.74, 499.70, 0.0, 180.0, 0.0]
SAFE_APPROACH = [825.41, 700.64, 499.70, 0.0, 180.0, 0.0]

class ChickenRobotMaster:
    def __init__(self, root):
        self.root = root
        self.server = None
        self.conn = None
        self.target_pos = None
        self.is_connected = False
        
        # 캘리브레이션 데이터 (유저님 제공 기반)
        self.M_pts = np.array([[-61.48, -146.0, 1653.0], [87.49, -10.75, 1558.11], 
                              [-408.0, 122.0, 1439.0], [-222.0, -307.0, 1070.0]], dtype=np.float32)
        self.R_pts = np.array([[860.92, 1073.92, 161.83], [1102.16, 1067.30, 154.97], 
                              [837.28, 630.72, 154.97], [837.28, 868.99, 679.61]], dtype=np.float32)
        res = cv2.estimateAffine3D(self.M_pts, self.R_pts)
        self.matrix = res[1] if len(res) == 3 else res[0]

        self.setup_ui()
        self.start_server()

    # [1] 서버/통신 로직
    def start_server(self):
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind(("0.0.0.0", PORT))
        self.server.listen(1)
        self.status_var.set("⏳ 로봇 접속 대기 중...")
        threading.Thread(target=self.accept_conn, daemon=True).start()

    def accept_conn(self):
        while True:
            try:
                conn, addr = self.server.accept()
                self.conn = conn
                self.is_connected = True
                self.status_var.set(f"✅ 접속됨: {addr[0]}")
                self.run_init_sequence()
                break
            except: time.sleep(1)

    def send_move(self, pos):
        if self.conn:
            try:
                msg = f"MOVE,{pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f},{pos[3]:.2f},{pos[4]:.2f},{pos[5]:.2f}\n"
                self.conn.sendall(msg.encode())
                data = self.conn.recv(1024).decode()
                return "DONE" in data
            except: return False
        return False

    # [2] 비전 워커 (카메라 화면 출력)
    def vision_worker(self):
        print("📸 카메라 초기화 중...")
        model = YOLO(r"C:\runs\detect\train12\weights\best.pt")
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        profile = pipeline.start(config)
        align = rs.align(rs.stream.color)
        intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        ROI = [401, 310, 241, 201]

        try:
            while True:
                frames = pipeline.wait_for_frames()
                aligned = align.process(frames)
                color_frame = aligned.get_color_frame()
                depth_frame = aligned.get_depth_frame()
                if not color_frame or not depth_frame: continue

                img = np.asanyarray(color_frame.get_data())
                crop_img = img[ROI[1]:ROI[1]+ROI[3], ROI[0]:ROI[0]+ROI[2]]
                results = model.predict(crop_img, conf=0.15, verbose=False)

                candidates = []
                for r in results:
                    for box in r.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        u, v = (x1 + x2) // 2 + ROI[0], (y1 + y2) // 2 + ROI[1]
                        depth = depth_frame.get_distance(u, v) * 1000
                        if depth > 0:
                            m_pt = rs.rs2_deproject_pixel_to_point(intr, [u, v], depth)
                            # 캘리브레이션 변환
                            src_pt = np.array([[[m_pt[0], m_pt[1], m_pt[2]]]], dtype=np.float32)
                            dst_pt = cv2.transform(src_pt, self.matrix)
                            rx, ry, rz = dst_pt[0][0]
                            candidates.append({'pos': [rx, ry, rz + Z_CORRECTION, 0, 180, 0], 'mz': m_pt[2], 'box': (x1+ROI[0], y1+ROI[1], x2+ROI[0], y2+ROI[1])})

                if candidates:
                    target = min(candidates, key=lambda x: x['mz'])
                    self.target_pos = target['pos']
                    # 화면 표시
                    bx1, by1, bx2, by2 = target['box']
                    cv2.rectangle(img, (bx1, by1), (bx2, by2), (0, 255, 255), 2)
                    cv2.putText(img, f"Target: {self.target_pos[0]:.1f}, {self.target_pos[1]:.1f}", (bx1, by1-10), 0, 0.6, (0,255,255), 2)

                cv2.rectangle(img, (ROI[0], ROI[1]), (ROI[0]+ROI[2], ROI[1]+ROI[3]), (0, 255, 0), 1)
                cv2.imshow("Chicken Monitor", img)
                if cv2.waitKey(1) == ord('q'): break
        finally:
            pipeline.stop()
            cv2.destroyAllWindows()

    def run_init_sequence(self):
        def task():
            self.send_move(SAFE_APPROACH)
            self.send_move(HOME_POS)
            self.status_var.set("🏠 홈 위치 대기 중")
            # 홈 복귀 후에 비전 시작
            threading.Thread(target=self.vision_worker, daemon=True).start()
        threading.Thread(target=task).start()

    def setup_ui(self):
        self.root.title("Chicken Master v2.1")
        self.root.geometry("450x300")
        self.status_var = tk.StringVar(value="서버 준비 중...")
        tk.Label(self.root, text="Robot Chicken Control", font=("Arial", 16, "bold")).pack(pady=15)
        tk.Label(self.root, textvariable=self.status_var, fg="blue", bg="#eee", width=40).pack(pady=10)
        tk.Button(self.root, text="🍗 치킨 잡기 실행", command=self.execute_pick, bg="orange", font=("Arial", 12, "bold"), height=2, width=20).pack(pady=20)

    def execute_pick(self):
        if self.target_pos and self.is_connected:
            def task():
                tx, ty, tz, rx, ry, rz = self.target_pos
                self.send_move(SAFE_APPROACH)
                self.send_move([tx, ty, SAFE_APPROACH[2], rx, ry, rz])
                time.sleep(2) # 상단 대기
                self.send_move([tx, ty, tz, rx, ry, rz]) # 집기 하강
            threading.Thread(target=task).start()
        else: messagebox.showwarning("경고", "치킨을 먼저 찾으세요.")

if __name__ == "__main__":
    root = tk.Tk()
    app = ChickenRobotMaster(root)
    root.mainloop()