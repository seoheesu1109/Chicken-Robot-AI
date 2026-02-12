import socket
import threading
import time
import tkinter as tk
from tkinter import scrolledtext
from PIL import Image, ImageTk
import numpy as np
import cv2
import pyrealsense2 as rs

# [1. 설정 데이터]
PORT = 30002
FIXED_ORI = [0.0, 180.0, 0.0] 
HOME_POS = [424.17, -69.51, 742.91] + FIXED_ORI
SAFE_APPROACH = [728.90, 947.73, 742.91] + FIXED_ORI
Y_OFFSET = -60.0 
Z_OFFSET = 35.0 

# [HSV 범위] - 어두운 벽면 방지
LOWER_CHICKEN = np.array([10, 70, 110])    
UPPER_CHICKEN = np.array([28, 255, 220]) 

class ChickenMasterV4:
    def __init__(self, root):
        self.root = root
        self.conn = None
        self.current_target = None
        
        # 캘리브레이션 데이터
        self.C_pts = np.array([[-60.99, -22.08, 751.00], [316.67, 13.45, 631.00], 
                               [-413.18, -43.74, 623.00], [-12.00, -276.04, 694.00], 
                               [-74.52, 253.81, 714.00]], dtype=np.float32)
        self.R_pts = np.array([[775.38, 893.70, 40.00], [410.45, 914.97, 178.26], 
                               [1142.18, 890.76, 174.17], [741.48, 641.75, 97.46], 
                               [789.80, 1178.78, 95.67]], dtype=np.float32)
        res = cv2.estimateAffine3D(self.C_pts, self.R_pts)
        self.matrix = res[1] if len(res) == 3 else res[0]

        self.setup_ui()
        threading.Thread(target=self.vision_worker, daemon=True).start()
        self.start_server()

    def setup_ui(self):
        self.root.title("Robot Chicken Control Center")
        self.root.geometry("800x980")
        self.root.configure(bg="#2c3e50")

        # 로그창
        self.log_widget = scrolledtext.ScrolledText(self.root, height=8, bg="#1E1E1E", fg="#00FF00")
        self.log_widget.pack(fill=tk.X, padx=10, pady=5)

        # 메인 버튼 프레임
        main_btn_frame = tk.Frame(self.root, bg="#2c3e50")
        main_btn_frame.pack(fill=tk.X, padx=20, pady=5)
        
        s = {"font": ("맑은 고딕", 10, "bold"), "height": 2, "width": 15}

        # 1행: 기본 이동 (Home, Safe)
        tk.Button(main_btn_frame, text="🏠 홈 이동", bg="#3498db", fg="white", command=self.go_home, **s).grid(row=0, column=0, padx=2, pady=2)
        tk.Button(main_btn_frame, text="🛡️ 안전위치", bg="#9b59b6", fg="white", command=self.go_safe, **s).grid(row=0, column=1, padx=2, pady=2)
        tk.Button(main_btn_frame, text="🔍 치킨 감지", bg="#f39c12", fg="white", command=self.scan_chicken, **s).grid(row=0, column=2, padx=2, pady=2)

        # 2행: 핵심 동작 (Down&Grip, Delivery)
        tk.Button(main_btn_frame, text="🍗 하강 후 그립", bg="#e74c3c", fg="white", command=self.down_and_grip, **s).grid(row=1, column=0, padx=2, pady=2)
        tk.Button(main_btn_frame, text="🚚 배출(배달)", bg="#2ecc71", fg="white", command=self.delivery_action, **s).grid(row=1, column=1, padx=2, pady=2)
        tk.Button(main_btn_frame, text="📍 타겟상공", bg="#e67e22", fg="white", command=self.go_target_top, **s).grid(row=1, column=2, padx=2, pady=2)

        # 3행: 수동 그리퍼 제어 (Grip, Release)
        tk.Button(main_btn_frame, text="✊ GRIP (잡기)", bg="#27ae60", fg="white", command=lambda: self.send_command("GRIP"), **s).grid(row=2, column=0, padx=2, pady=2)
        tk.Button(main_btn_frame, text="✋ RELEASE (놓기)", bg="#95a5a6", fg="white", command=lambda: self.send_command("RELEASE"), **s).grid(row=2, column=1, padx=2, pady=2)

        # XY 조절 프레임 (미세 조정용)
        xy_frame = tk.LabelFrame(self.root, text=" Target XY 미세 조절 (mm) ", bg="#2c3e50", fg="white")
        xy_frame.pack(fill=tk.X, padx=20, pady=5)
        
        tk.Button(xy_frame, text="X+", width=5, command=lambda: self.adjust_target(5, 0)).pack(side=tk.LEFT, padx=10, pady=5)
        tk.Button(xy_frame, text="X-", width=5, command=lambda: self.adjust_target(-5, 0)).pack(side=tk.LEFT, padx=10, pady=5)
        tk.Button(xy_frame, text="Y+", width=5, command=lambda: self.adjust_target(0, 5)).pack(side=tk.LEFT, padx=10, pady=5)
        tk.Button(xy_frame, text="Y-", width=5, command=lambda: self.adjust_target(0, -5)).pack(side=tk.LEFT, padx=10, pady=5)

        self.cam_label = tk.Label(self.root, bg="black"); self.cam_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    # --- 기능 함수들 ---
    def adjust_target(self, dx, dy):
        if self.current_target:
            self.current_target[0] += dx
            self.current_target[1] += dy
            self.add_log(f"📍 타겟 보정: X({self.current_target[0]:.1f}), Y({self.current_target[1]:.1f})")
        else:
            self.add_log("⚠️ 감지된 타겟이 없습니다.")

    def get_bottom_crop_coords(self, w, h):
        return (w//2 - 250), (h - 500), (w//2 + 250), h

    def is_valid(self, c, x1, y1, x2, y2):
        area = cv2.contourArea(c)
        if area < 800 or area > 20000: return False
        M = cv2.moments(c)
        if M["m00"] == 0: return False
        cx, cy = int(M["m10"]/M["m00"]) + x1, int(M["m01"]/M["m00"]) + y1
        if cx < x1+30 or cx > x2-30 or cy < y1+30 or cy > y2-30: return False
        return True

    def vision_worker(self):
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        cfg.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        self.pipeline.start(cfg)
        align = rs.align(rs.stream.color)
        try:
            while True:
                f = self.pipeline.wait_for_frames()
                a = align.process(f)
                color_f = a.get_color_frame()
                if not color_f: continue
                img = np.asanyarray(color_f.get_data())
                h, w = img.shape[:2]
                x1, y1, x2, y2 = self.get_bottom_crop_coords(w, h)
                crop = img[y1:y2, x1:x2]
                mask = cv2.inRange(cv2.cvtColor(crop, cv2.COLOR_BGR2HSV), LOWER_CHICKEN, UPPER_CHICKEN)
                cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                disp = img.copy()
                cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 255), 2)
                for c in cnts:
                    if self.is_valid(c, x1, y1, x2, y2):
                        cv2.drawContours(disp, [c + (x1, y1)], -1, (0, 255, 0), 2)
                self.root.after(0, self.update_cam_image, disp)
        finally: self.pipeline.stop()

    def scan_chicken(self):
        self.add_log("🔍 스캔 중...")
        f = self.pipeline.wait_for_frames()
        a = rs.align(rs.stream.color).process(f)
        color_f, depth_f = a.get_color_frame(), a.get_depth_frame()
        if not color_f or not depth_f: return
        img = np.asanyarray(color_f.get_data())
        h, w = img.shape[:2]
        x1, y1, x2, y2 = self.get_bottom_crop_coords(w, h)
        mask = cv2.inRange(cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2HSV), LOWER_CHICKEN, UPPER_CHICKEN)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid = [c for c in cnts if self.is_valid(c, x1, y1, x2, y2)]
        if valid:
            valid.sort(key=cv2.contourArea, reverse=True)
            M = cv2.moments(valid[0])
            u, v = int(M["m10"]/M["m00"]) + x1, int(M["m01"]/M["m00"]) + y1
            d = depth_f.get_distance(u, v) * 1000
            intr = self.pipeline.get_active_profile().get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
            p_cam = rs.rs2_deproject_pixel_to_point(intr, [u, v], d)
            dst = cv2.transform(np.array([[[p_cam[0], p_cam[1], p_cam[2]]]], dtype=np.float32), self.matrix)
            self.current_target = [dst[0][0][0], dst[0][0][1] + Y_OFFSET, dst[0][0][2] + Z_OFFSET, 0, 180, 0]
            self.add_log("✅ 타겟 감지 완료")
        else: self.add_log("❌ 감지 실패")

    # --- 로봇 이동 명령 ---
    def get_robot_current_y(self):
        if not self.conn: return 0.0
        try:
            self.conn.sendall(b"GET\n")
            resp = self.conn.recv(1024).decode().strip()
            if resp:
                coords = [float(x) for x in resp.split(',')]
                return coords[1]
        except: return 0.0
    def send_command_and_wait(self, cmd):
        if not self.conn: return False
        try:
            self.add_log(f"➡️ {cmd}")
            # 명령 전송 전 버퍼를 비우는 작업 (선택사항이나 권장)
            self.conn.setblocking(False)
            try:
                while self.conn.recv(1024): pass
            except: pass
            self.conn.setblocking(True)

            self.conn.sendall((cmd + "\n").encode())
            
            # 응답 수신
            raw_data = self.conn.recv(1024).decode().strip()
            
            # 개선된 체크: 데이터 안에 "DONE"이 포함되어 있는지 확인
            if "DONE" in raw_data:
                self.add_log(f"✅ 로봇 완료 수신")
                return True
            elif "ERROR" in raw_data:
                self.add_log(f"❌ 로봇 측 실행 에러")
                return False
            else:
                self.add_log(f"⚠️ 예상치 못한 응답: {raw_data}")
                # 만약 응답에 DONE이 없더라도 로그에 찍힌 것처럼 데이터가 왔다면 
                # 일단 다음 진행을 위해 True를 리턴하게 할 수도 있습니다.
                return False 
        except Exception as e:
            self.add_log(f"❌ 통신 오류: {e}")
            return False
    def go_home(self):
        cy = self.get_robot_current_y()
        if cy >= 190:
            self.send_command(f"MOVE,{SAFE_APPROACH[0]},{SAFE_APPROACH[1]},{SAFE_APPROACH[2]},{SAFE_APPROACH[3]},{SAFE_APPROACH[4]},{SAFE_APPROACH[5]}")
            time.sleep(0.2)
        self.send_command(f"MOVE,{HOME_POS[0]},{HOME_POS[1]},{HOME_POS[2]},{HOME_POS[3]},{HOME_POS[4]},{HOME_POS[5]}")
    def go_safe(self): self.send_command(f"MOVE,{SAFE_APPROACH[0]},{SAFE_APPROACH[1]},{SAFE_APPROACH[2]},{SAFE_APPROACH[3]},{SAFE_APPROACH[4]},{SAFE_APPROACH[5]}")
    def go_target_top(self):
        if self.current_target: self.send_command(f"MOVE,{self.current_target[0]:.2f},{self.current_target[1]:.2f},{SAFE_APPROACH[2]},{FIXED_ORI[0]},{FIXED_ORI[1]},{FIXED_ORI[2]}")
    def down_and_grip(self):
        if not self.current_target:
            self.add_log("⚠️ 타겟이 없습니다.")
            return
        # 별도 스레드에서 로직 실행 (UI 프리징 방지)
        threading.Thread(target=self._execute_down_grip, daemon=True).start()

    def _execute_down_grip(self):
        tx, ty, tz, rx, ry, rz = self.current_target
        
        # 1. 하강 명령 전송
        if self.send_command_and_wait(f"DOWN,{tx:.2f},{ty:.2f},{tz:.2f},{rx},{ry},{rz}"):
            time.sleep(0.1)
            # 2. 하강 완료(DONE 수신) 후 그립 명령 전송
            if self.send_command_and_wait("GRIP"):
                time.sleep(0.1)
                # 3. 그립 완료 후 다시 안전 상공으로 이동
                self.send_command_and_wait(f"MOVE,{tx:.2f},{ty:.2f},{SAFE_APPROACH[2]},{rx},{ry},{rz}")
                self.add_log("🍗 하강 및 그립 작업 완료!")
    def delivery_action(self):
        self.send_command(f"MOVE,{HOME_POS[0]},{HOME_POS[1]},{HOME_POS[2]},{HOME_POS[3]},{HOME_POS[4]},{HOME_POS[5]}")
        time.sleep(0.1); self.send_command("RELEASE")

    def update_cam_image(self, frame):
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).resize((640, 360))
        itk = ImageTk.PhotoImage(image=img); self.cam_label.itk = itk; self.cam_label.configure(image=itk)
    def add_log(self, msg):
        self.log_widget.configure(state='normal'); self.log_widget.insert(tk.END, f"[{time.strftime('%H:%M:%S')}] {msg}\n"); self.log_widget.see(tk.END); self.log_widget.configure(state='disabled')
    def start_server(self):
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM); self.server.bind(("0.0.0.0", PORT)); self.server.listen(1)
        threading.Thread(target=self.accept_conn, daemon=True).start()
    def accept_conn(self):
        while True: self.conn, _ = self.server.accept(); self.add_log("✅ 로봇 연결됨")
    def send_command(self, cmd):
        if self.conn:
            try: self.add_log(f"➡️ {cmd}"); self.conn.sendall((cmd + "\n").encode()); return True
            except: return False
        return False

if __name__ == "__main__":
    root = tk.Tk(); app = ChickenMasterV4(root); root.mainloop()

    