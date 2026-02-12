import socket
import threading
import time
import tkinter as tk
from tkinter import messagebox
import numpy as np
import cv2
import pyrealsense2 as rs
from ultralytics import YOLO

# [설정 데이터 & 실측 보정값 적용]
PORT = 30002

# 1. X 보정: 기존 -25.86 + 차이 28.18 = 2.32
X_OFFSET = 2.32

# 2. Y 보정: 기존 -50.0 + 차이 97.85 = 47.85
Y_OFFSET_CORRECTION = 47.85

# 3. Z 보정: 기존 30.0 + 차이 -23.70 = 6.30
Z_CORRECTION = 6.30

HOME_POS = [647.92, -119.74, 499.70, 0.0, 180.0, 0.0]
SAFE_APPROACH = [825.41, 700.64, 499.70, 0.0, 180.0, 0.0]
DROP_POS = [1262.44, -177.70, -308.44, 0.0, 180.0, 0.0]

class ChickenRobotMaster:
    def __init__(self, root):
        self.root = root
        self.conn = None
        self.is_connected = False
        self.target_pos = None
        
        # 캘리브레이션 행렬
        self.M_pts = np.array([[-61.48, -146.0, 1653.0], [87.49, -10.75, 1558.11], 
                              [-408.0, 122.0, 1439.0], [-222.0, -307.0, 1070.0]], dtype=np.float32)
        self.R_pts = np.array([[860.92, 1073.92, 161.83], [1102.16, 1067.30, 154.97], 
                              [837.28, 630.72, 154.97], [837.28, 868.99, 679.61]], dtype=np.float32)
        res = cv2.estimateAffine3D(self.M_pts, self.R_pts)
        self.matrix = res[1] if len(res) == 3 else res[0]

        self.setup_ui() 
        print("📸 카메라 초기화 및 실측 보정값 적용 완료")
        threading.Thread(target=self.vision_worker, daemon=True).start()
        self.start_server()

    def setup_ui(self):
        self.root.title("Chicken Picking Master")
        self.root.geometry("450x350")
        self.status_var = tk.StringVar(value="📷 카메라 기동 중...")
        
        tk.Label(self.root, text="Chicken Robot Control", font=("Arial", 16, "bold")).pack(pady=20)
        tk.Label(self.root, textvariable=self.status_var, bg="#333", fg="white", width=40, height=2).pack()
        
        self.btn_pick = tk.Button(self.root, text="🍗 치킨 잡기 실행", command=self.execute_pick, 
                                  bg="orange", font=("Arial", 12, "bold"), height=2, width=20)
        self.btn_pick.pack(pady=30)

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
                
                time.sleep(1.0) 
                self.clear_buffer()
                self.conn.sendall(b"GET\n")
                raw = self.conn.recv(1024).decode().strip()
                
                parts = raw.split(',')
                curr_y = float(parts[2]) if len(parts) > 2 else 0.0
                self.run_initial_sequence(curr_y)
                break
            except Exception as e: 
                time.sleep(1)

    def clear_buffer(self):
        self.conn.setblocking(False)
        try:
            while True:
                if not self.conn.recv(1024): break
        except: pass
        self.conn.setblocking(True)

    def send_gripper(self, cmd):
        if not self.conn: return False
        try:
            self.clear_buffer()
            self.conn.sendall(f"{cmd}\n".encode())
            self.conn.recv(1024) 
            return True
        except: return False

    def send_move(self, pos):
        if not self.conn: return False
        try:
            self.clear_buffer()
            msg = f"MOVE,{pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f},{pos[3]:.2f},{pos[4]:.2f},{pos[5]:.2f}\n"
            self.conn.sendall(msg.encode())
            resp = self.conn.recv(1024).decode().strip()
            return "DONE" in resp
        except: return False

    def run_initial_sequence(self, curr_y):
        def task():
            self.send_gripper("RELEASE")
            time.sleep(0.5)
            if curr_y > 190:
                self.send_move(SAFE_APPROACH)
            self.send_move(HOME_POS)
            self.status_var.set("🏠 작업 준비 완료")
        threading.Thread(target=task, daemon=True).start()

    def vision_worker(self):
        try:
            model = YOLO(r"C:\runs\detect\train12\weights\best.pt")
            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
            config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
            profile = pipeline.start(config)
            align = rs.align(rs.stream.color)
            intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
            ROI = [401, 310, 241, 201]

            while True:
                frames = pipeline.wait_for_frames()
                aligned = align.process(frames)
                color_frame = aligned.get_color_frame()
                depth_frame = aligned.get_depth_frame()
                if not color_frame or not depth_frame: continue

                img = np.asanyarray(color_frame.get_data())
                crop = img[ROI[1]:ROI[1]+ROI[3], ROI[0]:ROI[0]+ROI[2]]
                results = model.predict(crop, conf=0.25, verbose=False)

                candidates = []
                for r in results:
                    for box in r.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        u, v = (x1 + x2) // 2 + ROI[0], (y1 + y2) // 2 + ROI[1]
                        depth = depth_frame.get_distance(u, v) * 1000
                        if depth > 0:
                            m_pt = rs.rs2_deproject_pixel_to_point(intr, [u, v], depth)
                            src_pt = np.array([[[m_pt[0], m_pt[1], m_pt[2]]]], dtype=np.float32)
                            dst_pt = cv2.transform(src_pt, self.matrix)
                            
                            # 보정값이 반영된 최종 좌표 계산
                            rx = dst_pt[0][0][0] + X_OFFSET
                            ry = dst_pt[0][0][1] + Y_OFFSET_CORRECTION
                            rz = dst_pt[0][0][2] + Z_CORRECTION
                            
                            candidates.append({
                                'pos': [rx, ry, rz, 0, 180, 0], 
                                'mz': m_pt[2], 
                                'box': (x1+ROI[0], y1+ROI[1], x2+ROI[0], y2+ROI[1])
                            })

                if candidates:
                    target = min(candidates, key=lambda x: x['mz'])
                    self.target_pos = target['pos']
                    
                    # 터미널 출력 (보정 후 좌표 확인용)
                    print(f"🎯 보정 완료 좌표 -> X: {self.target_pos[0]:.2f}, Y: {self.target_pos[1]:.2f}, Z: {self.target_pos[2]:.2f}")
                    
                    bx1, by1, bx2, by2 = target['box']
                    cv2.rectangle(img, (bx1, by1), (bx2, by2), (0, 255, 255), 2)
                
                cv2.rectangle(img, (ROI[0], ROI[1]), (ROI[0]+ROI[2], ROI[1]+ROI[3]), (0, 255, 0), 1)
                cv2.imshow("Chicken Monitor", img)
                if cv2.waitKey(1) == ord('q'): break
        finally:
            pipeline.stop()
            cv2.destroyAllWindows()

    def execute_pick(self):
        if self.target_pos and self.is_connected:
            self.btn_pick.config(state=tk.DISABLED) 
            def task():
                tx, ty, tz, rx, ry, rz = self.target_pos
                self.status_var.set(f"🚀 이동 중: X{tx:.1f} Y{ty:.1f}")
                
                self.send_move(SAFE_APPROACH)
                # 하강 및 잡기 동작
                if self.send_move([tx, ty, SAFE_APPROACH[2], rx, ry, rz]):
                    if self.send_move([tx, ty, tz, rx, ry, rz]):
                        time.sleep(0.5)
                        self.send_gripper("GRIP")
                        time.sleep(1.0)
                        self.send_move([tx, ty, SAFE_APPROACH[2], rx, ry, rz])

                self.send_move(HOME_POS)
                self.send_move([DROP_POS[0], DROP_POS[1], SAFE_APPROACH[2], 0, 180, 0])
                if self.send_move(DROP_POS):
                    time.sleep(0.5)
                    self.send_gripper("RELEASE")
                    time.sleep(1.0)
                    self.send_move([DROP_POS[0], DROP_POS[1], SAFE_APPROACH[2], 0, 180, 0])

                self.send_move(HOME_POS)
                self.status_var.set("✅ 작업 완료")
                self.root.after(0, lambda: self.btn_pick.config(state=tk.NORMAL))

            threading.Thread(target=task, daemon=True).start()
        else: 
            messagebox.showwarning("경고", "로봇 연결 확인 및 치킨 감지가 필요합니다.")

if __name__ == "__main__":
    root = tk.Tk()
    app = ChickenRobotMaster(root)
    root.mainloop()