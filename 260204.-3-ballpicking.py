import socket
import threading
import time
import tkinter as tk
from tkinter import messagebox, scrolledtext
import numpy as np
import cv2
import pyrealsense2 as rs

# [1. 설정 데이터]
PORT = 30002
FIXED_ORI = [0.0, 180.0, 0.0] 
HOME_POS = [424.17, -69.51, 742.91] + FIXED_ORI
SAFE_APPROACH = [728.90, 947.73, 742.91] + FIXED_ORI

# --- [강제 보정값 설정] ---
Y_OFFSET = -60.0  
# -----------------------

class BallPickupMaster:
    def __init__(self, root):
        self.root = root
        self.conn = None
        self.is_connected = False
        
        # 캘리브레이션 데이터
        self.C_pts = np.array([
            [-60.99, -22.08, 751.00], [316.67, 13.45, 631.00], 
            [-413.18, -43.74, 623.00], [-12.00, -276.04, 694.00], 
            [-74.52, 253.81, 714.00]], dtype=np.float32)
        self.R_pts = np.array([
            [775.38, 893.70, 40.00], [410.45, 914.97, 178.26], 
            [1142.18, 890.76, 174.17], [741.48, 641.75, 97.46], 
            [789.80, 1178.78, 95.67]], dtype=np.float32)
        
        res = cv2.estimateAffine3D(self.C_pts, self.R_pts)
        self.matrix = res[1] if len(res) == 3 else res[0]

        self.setup_ui()
        self.add_log(f"🚀 시스템 시작 (Y 보정: {Y_OFFSET}mm)")
        
        threading.Thread(target=self.vision_worker, daemon=True).start()
        self.start_server()

    def setup_ui(self):
        self.root.title("Ball Pickup Master (Smart Pathing)")
        self.root.geometry("600x650")
        self.status_var = tk.StringVar(value="📷 카메라 대기 중...")
        tk.Label(self.root, textvariable=self.status_var, bg="#333", fg="white", font=("Arial", 12)).pack(fill=tk.X)
        self.log_widget = scrolledtext.ScrolledText(self.root, height=18, state='disabled', bg="#f8f8f8")
        self.log_widget.pack(fill=tk.BOTH, padx=10, expand=True)
        self.btn_pick = tk.Button(self.root, text="🔵 (릴리즈/경로보정)", command=self.execute_pick, 
                                  bg="#27AE60", fg="white", font=("Arial", 14, "bold"), height=2)
        self.btn_pick.pack(pady=20, fill=tk.X, padx=50)

    def add_log(self, msg):
        timestamp = time.strftime("[%H:%M:%S] ")
        self.log_widget.configure(state='normal')
        self.log_widget.insert(tk.END, timestamp + msg + "\n")
        self.log_widget.see(tk.END)
        self.log_widget.configure(state='disabled')

    def start_server(self):
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind(("0.0.0.0", PORT))
        self.server.listen(1)
        threading.Thread(target=self.accept_conn, daemon=True).start()

    def accept_conn(self):
        while True:
            conn, addr = self.server.accept()
            self.conn = conn
            self.is_connected = True
            self.add_log(f"✅ 로봇 연결: {addr[0]}")
            self.send_command("RELEASE")
            self.send_command(f"MOVE,{HOME_POS[0]},{HOME_POS[1]},{HOME_POS[2]},{HOME_POS[3]},{HOME_POS[4]},{HOME_POS[5]}")
            break

    def send_command(self, cmd_str):
        if not self.conn: return False
        try:
            self.conn.setblocking(False)
            try: 
                while self.conn.recv(1024): pass
            except: pass
            self.conn.setblocking(True)
            self.add_log(f"➡️ 전송: {cmd_str}")
            self.conn.sendall((cmd_str + "\n").encode())
            resp = self.conn.recv(1024).decode().strip()
            self.add_log(f"⬅️ 수신: {resp}")
            return "DONE" in resp
        except: return False

    def vision_worker(self):
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        profile = self.pipeline.start(config)
        self.align = rs.align(rs.stream.color)
        self.intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        while True:
            frames = self.pipeline.wait_for_frames()
            color_frame = self.align.process(frames).get_color_frame()
            if not color_frame: continue
            img = np.asanyarray(color_frame.get_data())
            cv2.imshow("Monitor", img)
            if cv2.waitKey(1) == ord('q'): break
        self.pipeline.stop()

    def get_target_after_stop(self):
        self.add_log("📸 정밀 인식 및 보정 계산...")
        for _ in range(15): self.pipeline.wait_for_frames()
        frames = self.pipeline.wait_for_frames()
        aligned = self.align.process(frames)
        color_f = aligned.get_color_frame()
        depth_f = aligned.get_depth_frame()
        if not color_f or not depth_f: return None
        img = np.asanyarray(color_f.get_data())
        hsv = cv2.cvtColor(cv2.GaussianBlur(img, (11, 11), 0), cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([85, 100, 100]), np.array([105, 255, 255]))
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(cnts) > 0:
            c = max(cnts, key=cv2.contourArea)
            ((u, v), radius) = cv2.minEnclosingCircle(c)
            if radius > 15:
                depth = depth_f.get_distance(int(u), int(v)) * 1000
                if depth > 0:
                    p_cam = rs.rs2_deproject_pixel_to_point(self.intrinsics, [u, v], depth)
                    src_pt = np.array([[[p_cam[0], p_cam[1], p_cam[2]]]], dtype=np.float32)
                    dst_pt = cv2.transform(src_pt, self.matrix)
                    return [dst_pt[0][0][0], dst_pt[0][0][1] + Y_OFFSET, dst_pt[0][0][2], 0, 180, 0]
        return None

    def execute_pick(self):
        if not self.is_connected: return
        self.btn_pick.config(state=tk.DISABLED)

        def task():
            # 1. 안전 위치 이동
            self.send_command(f"MOVE,{SAFE_APPROACH[0]},{SAFE_APPROACH[1]},{SAFE_APPROACH[2]},{SAFE_APPROACH[3]},{SAFE_APPROACH[4]},{SAFE_APPROACH[5]}")
            time.sleep(1.2)
            
            # 2. 타겟 획득
            target = self.get_target_after_stop()
            
            if target:
                tx, ty, tz, rx, ry, rz = target
                self.add_log(f"🎯 타겟 확정 (Y보정포함): Y={ty:.1f}")
                
                # 3. 이동 및 그리핑
                self.send_command(f"MOVE,{tx:.2f},{ty:.2f},{SAFE_APPROACH[2]},{rx},{ry},{rz}")
                self.send_command(f"DOWN,{tx:.2f},{ty:.2f},{tz:.2f},{rx},{ry},{rz}")
                self.send_command("GRIP")
                
                # 4. 상승
                self.send_command(f"MOVE,{tx:.2f},{ty:.2f},{SAFE_APPROACH[2]},{rx},{ry},{rz}")
                
                # 5. 홈으로 복귀 (조건부 경로)
                # 현재 공 위치(ty)가 190 이상이면 안전 위치를 거쳐감
                if ty >= 190:
                    self.add_log(f"⚠️ Y({ty:.1f})가 190 이상입니다. 안전위치 경유.")
                    self.send_command(f"MOVE,{SAFE_APPROACH[0]},{SAFE_APPROACH[1]},{SAFE_APPROACH[2]},{SAFE_APPROACH[3]},{SAFE_APPROACH[4]},{SAFE_APPROACH[5]}")
                
                # 홈 위치 도착 및 릴리즈
                self.send_command(f"MOVE,{HOME_POS[0]},{HOME_POS[1]},{HOME_POS[2]},{HOME_POS[3]},{HOME_POS[4]},{HOME_POS[5]}")
                self.send_command("RELEASE")
                self.add_log("🏁 시퀀스 완료 및 릴리즈")
            else:
                self.add_log("❌ 공 인식 실패로 중단")
                self.send_command(f"MOVE,{HOME_POS[0]},{HOME_POS[1]},{HOME_POS[2]},{HOME_POS[3]},{HOME_POS[4]},{HOME_POS[5]}")

            self.root.after(0, lambda: self.btn_pick.config(state=tk.NORMAL))

        threading.Thread(target=task, daemon=True).start()

if __name__ == "__main__":
    root = tk.Tk()
    app = BallPickupMaster(root)
    root.mainloop()