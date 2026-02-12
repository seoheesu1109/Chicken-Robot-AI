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
Y_OFFSET = -60.0  

class BallPickupDetailControl:
    def __init__(self, root):
        self.root = root
        self.conn = None
        self.is_connected = False
        self.current_target = None # 인식된 타겟 좌표 저장용
        
        # 캘리브레이션 데이터 및 행렬 계산
        self.C_pts = np.array([[-60.99, -22.08, 751.00], [316.67, 13.45, 631.00], [-413.18, -43.74, 623.00], [-12.00, -276.04, 694.00], [-74.52, 253.81, 714.00]], dtype=np.float32)
        self.R_pts = np.array([[775.38, 893.70, 40.00], [410.45, 914.97, 178.26], [1142.18, 890.76, 174.17], [741.48, 641.75, 97.46], [789.80, 1178.78, 95.67]], dtype=np.float32)
        res = cv2.estimateAffine3D(self.C_pts, self.R_pts)
        self.matrix = res[1] if len(res) == 3 else res[0]

        self.setup_ui()
        threading.Thread(target=self.vision_worker, daemon=True).start()
        self.start_server()

    def setup_ui(self):
        self.root.title("Ball Robot Manual Control")
        self.root.geometry("600x750")
        
        # 로그 창
        self.log_widget = scrolledtext.ScrolledText(self.root, height=15, bg="#f0f0f0")
        self.log_widget.pack(fill=tk.BOTH, padx=10, pady=10)

        # --- 버튼 프레임 ---
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(fill=tk.BOTH, padx=20, pady=10)

        # 버튼 스타일
        s = {"font": ("Arial", 11, "bold"), "height": 2}

        # 1. 홈 이동 / 안전위치
        tk.Button(btn_frame, text="🏠 홈 이동", bg="#3498db", fg="white", command=self.go_home, **s).grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        tk.Button(btn_frame, text="🛡️ 안전위치", bg="#9b59b6", fg="white", command=self.go_safe, **s).grid(row=0, column=1, sticky="nsew", padx=5, pady=5)

        # 2. XY 보정 (인식) / 타겟상공 이동
        tk.Button(btn_frame, text="🔍 XY 보정 (인식)", bg="#f1c40f", command=self.scan_target, **s).grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        tk.Button(btn_frame, text="📍 타겟상공 이동", bg="#e67e22", fg="white", command=self.go_target_top, **s).grid(row=1, column=1, sticky="nsew", padx=5, pady=5)

        # 3. 하강 후 그립 / 배송
        tk.Button(btn_frame, text="👇 하강 후 그립", bg="#e74c3c", fg="white", command=self.down_and_grip, **s).grid(row=2, column=0, sticky="nsew", padx=5, pady=5)
        tk.Button(btn_frame, text="🚚 배송 (안전경유)", bg="#2ecc71", fg="white", command=self.delivery_path, **s).grid(row=2, column=1, sticky="nsew", padx=5, pady=5)

        # 그리퍼 수동
        tk.Button(btn_frame, text="🔓 RELEASE", bg="#bdc3c7", command=lambda: self.send_command("RELEASE"), height=1).grid(row=3, column=0, sticky="nsew", padx=5, pady=10)
        tk.Button(btn_frame, text="🔒 GRIP", bg="#bdc3c7", command=lambda: self.send_command("GRIP"), height=1).grid(row=3, column=1, sticky="nsew", padx=5, pady=10)

        for i in range(2): btn_frame.grid_columnconfigure(i, weight=1)

    # --- [ 개별 동작 함수 ] ---
    def get_robot_current_y(self):
        """로봇에게 현재 좌표를 물어보고 Y값을 반환함"""
        if not self.conn: return 0.0
        try:
            # 1. 로봇에게 GET 명령 전송 (DRL의 'GET' 섹션 실행)
            self.conn.sendall(b"GET\n")
            # 2. 로봇으로부터 좌표 수신 (예: "424.17,-69.51,742.91,0.00,180.00,0.00")
            resp = self.conn.recv(1024).decode().strip()
            if resp:
                coords = [float(x) for x in resp.split(',')]
                return coords[1]  # Y값은 인덱스 1번
        except Exception as e:
            self.add_log(f"⚠️ 좌표 획득 실패: {e}")
        return 0.0

    def go_home(self):
        # 로봇에게 실시간 Y좌표를 물어봄
        current_y = self.get_robot_current_y()
        self.add_log(f"📍 로봇 현재 Y: {current_y:.2f}")

        # Y값이 190 이상이면 안전위치 경유
        if current_y >= 190:
            self.add_log(f"⚠️ Y가 190 이상({current_y:.1f})이므로 안전위치를 경유합니다.")
            self.send_command(f"MOVE,{SAFE_APPROACH[0]},{SAFE_APPROACH[1]},{SAFE_APPROACH[2]},{SAFE_APPROACH[3]},{SAFE_APPROACH[4]},{SAFE_APPROACH[5]}")
        
        # 홈 위치로 이동
        self.send_command(f"MOVE,{HOME_POS[0]},{HOME_POS[1]},{HOME_POS[2]},{HOME_POS[3]},{HOME_POS[4]},{HOME_POS[5]}")
        
        # 2. 홈 위치로 이동
        self.send_command(f"MOVE,{HOME_POS[0]},{HOME_POS[1]},{HOME_POS[2]},{HOME_POS[3]},{HOME_POS[4]},{HOME_POS[5]}")

    def go_safe(self):
        self.send_command(f"MOVE,{SAFE_APPROACH[0]},{SAFE_APPROACH[1]},{SAFE_APPROACH[2]},{SAFE_APPROACH[3]},{SAFE_APPROACH[4]},{SAFE_APPROACH[5]}")

    def scan_target(self):
        """기존 HSV 로직으로 타겟 좌표 계산만 수행"""
        target = self.get_target_after_stop()
        if target:
            self.current_target = target
            self.add_log(f"✅ 인식 성공: X={target[0]:.1f}, Y={target[1]:.1f}")
        else:
            self.add_log("❌ 인식 실패 (공이 보이지 않음)")

    def go_target_top(self):
        if not self.current_target:
            messagebox.showwarning("알림", "먼저 'XY 보정'을 실행하세요.")
            return
        tx, ty, _, rx, ry, rz = self.current_target
        # 안전한 Z높이(SAFE_APPROACH[2]) 유지하며 타겟 위로 이동
        self.send_command(f"MOVE,{tx:.2f},{ty:.2f},{SAFE_APPROACH[2]},{rx},{ry},{rz}")

    def down_and_grip(self):
        if not self.current_target: return
        tx, ty, tz, rx, ry, rz = self.current_target
        self.send_command(f"DOWN,{tx:.2f},{ty:.2f},{tz:.2f},{rx},{ry},{rz}")
        time.sleep(0.3)
        self.send_command("GRIP")
        time.sleep(0.3)
        self.send_command(f"MOVE,{tx:.2f},{ty:.2f},{SAFE_APPROACH[2]},{rx},{ry},{rz}") # 다시 상승

    def delivery_path(self):
        self.add_log("🚚 배송 경로 가동 (안전경유)")
        # 반드시 안전위치 경유
        #self.send_command(f"MOVE,{SAFE_APPROACH[0]},{SAFE_APPROACH[1]},{SAFE_APPROACH[2]},{SAFE_APPROACH[3]},{SAFE_APPROACH[4]},{SAFE_APPROACH[5]}")
        # 홈 이동 및 릴리즈
        self.send_command(f"MOVE,{HOME_POS[0]},{HOME_POS[1]},{HOME_POS[2]},{HOME_POS[3]},{HOME_POS[4]},{HOME_POS[5]}")
        self.send_command("RELEASE")

    # --- [ 통신 및 비전 기본 로직 ] ---
    def add_log(self, msg):
        self.log_widget.configure(state='normal')
        self.log_widget.insert(tk.END, f"[{time.strftime('%H:%M:%S')}] {msg}\n")
        self.log_widget.see(tk.END)
        self.log_widget.configure(state='disabled')

    def start_server(self):
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.bind(("0.0.0.0", PORT))
        self.server.listen(1)
        threading.Thread(target=self.accept_conn, daemon=True).start()

    def accept_conn(self):
        while True:
            conn, _ = self.server.accept()
            self.conn = conn
            self.is_connected = True
            self.add_log("✅ 로봇 연결됨")
            self.go_home()
            break

    def send_command(self, cmd_str):
        if not self.conn: return False
        try:
            self.add_log(f"➡️ {cmd_str}")
            self.conn.sendall((cmd_str + "\n").encode())
            resp = self.conn.recv(1024).decode().strip()
            self.add_log(f"⬅️ {resp}")
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
            cv2.imshow("HSV Monitor", img)
            if cv2.waitKey(1) == ord('q'): break

    def get_target_after_stop(self):
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
                    dst_pt = cv2.transform(np.array([[[p_cam[0], p_cam[1], p_cam[2]]]], dtype=np.float32), self.matrix)
                    return [dst_pt[0][0][0], dst_pt[0][0][1] + Y_OFFSET, dst_pt[0][0][2], 0, 180, 0]
        return None

if __name__ == "__main__":
    root = tk.Tk()
    app = BallPickupDetailControl(root)
    root.mainloop()