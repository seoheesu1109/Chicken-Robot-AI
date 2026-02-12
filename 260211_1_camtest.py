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
Y_OFFSET = 60.0  # 로봇 좌표계 보정값
Z_FINE_TUNE = 0

# [HSV 범위 - 치킨 색상]
LOWER_CHICKEN = np.array([10, 70, 110])    
UPPER_CHICKEN = np.array([28, 255, 220])

class ChickenMasterV4:
    def __init__(self, root):
        self.root = root
        self.conn = None
        self.pipeline = None
        self.current_target = None 
        self.auto_running = False
        
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
        self.root.title("Chicken Robot Master - Smart Segmenter")
        self.root.geometry("800x980")
        self.root.configure(bg="#2c3e50")

        self.log_widget = scrolledtext.ScrolledText(self.root, height=8, bg="#1E1E1E", fg="#00FF00")
        self.log_widget.pack(fill=tk.X, padx=10, pady=5)

        main_btn_frame = tk.Frame(self.root, bg="#2c3e50")
        main_btn_frame.pack(fill=tk.X, padx=20, pady=5)
        
        s = {"font": ("맑은 고딕", 10, "bold"), "height": 2, "width": 15}

        tk.Button(main_btn_frame, text="🏠 홈 이동", bg="#3498db", fg="white", command=self.go_home, **s).grid(row=0, column=0, padx=2, pady=2)
        tk.Button(main_btn_frame, text="🛡️ 안전위치", bg="#9b59b6", fg="white", command=self.go_safe, **s).grid(row=0, column=1, padx=2, pady=2)
        tk.Button(main_btn_frame, text="🔍 낱개 감지", bg="#f39c12", fg="white", command=self.scan_chicken, **s).grid(row=0, column=2, padx=2, pady=2)

        tk.Button(main_btn_frame, text="🍗 하강 후 그립", bg="#e74c3c", fg="white", command=self.down_and_grip, **s).grid(row=1, column=0, padx=2, pady=2)
        tk.Button(main_btn_frame, text="🚚 배출(배달)", bg="#2ecc71", fg="white", command=self.delivery_action, **s).grid(row=1, column=1, padx=2, pady=2)
        tk.Button(main_btn_frame, text="📍 타겟상공", bg="#e67e22", fg="white", command=self.go_target_top, **s).grid(row=1, column=2, padx=2, pady=2)

        tk.Button(main_btn_frame, text="✊ GRIP", bg="#27ae60", fg="white", command=lambda: self.send_command("GRIP"), **s).grid(row=2, column=0, padx=2, pady=2)
        tk.Button(main_btn_frame, text="✋ RELEASE", bg="#95a5a6", fg="white", command=lambda: self.send_command("RELEASE"), **s).grid(row=2, column=1, padx=2, pady=2)

        self.btn_auto_start = tk.Button(main_btn_frame, text="🤖 풀오토 시작", bg="#f1c40f", fg="black", command=self.start_full_auto, **s)
        self.btn_auto_start.grid(row=3, column=0, padx=2, pady=5)
        
        self.btn_auto_stop = tk.Button(main_btn_frame, text="🛑 무한루프 중단", bg="#e67e22", fg="white", command=self.stop_full_auto, **s)
        self.btn_auto_stop.grid(row=3, column=1, padx=2, pady=5)

        self.cam_label = tk.Label(self.root, bg="black")
        self.cam_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    def scan_chicken(self):
        """[업데이트] 감지된 조각 중 카메라와 가장 가까운(가장 높은) 조각을 선택"""
        self.add_log("🔍 가장 가까운 타겟 분석 중...")
        f = self.pipeline.wait_for_frames()
        a = rs.align(rs.stream.color).process(f)
        color_f, depth_f = a.get_color_frame(), a.get_depth_frame()
        if not color_f or not depth_f: return
        
        img = np.asanyarray(color_f.get_data())
        h, w = img.shape[:2]
        x1, x2 = w // 2 - 300, w // 2 + 100
        y1, y2 = h // 2 - 100, h - 100
        crop = img[y1:y2, x1:x2]
        
        # 1. 낱개 분리를 위한 전처리
        blurred = cv2.GaussianBlur(crop, (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, LOWER_CHICKEN, UPPER_CHICKEN)
        
        # 2. 덩어리 분리 알고리즘 (Distance Transform)
        kernel = np.ones((5, 5), np.uint8)
        mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        dist_transform = cv2.distanceTransform(mask_closed, cv2.DIST_L2, 5)
        
        # 확실한 중심부만 남김 (0.35 비율)
        _, sure_fg = cv2.threshold(dist_transform, 0.35 * dist_transform.max() if dist_transform.max() > 0 else 0, 255, 0)
        sure_fg = np.uint8(sure_fg)
        
        cnts, _ = cv2.findContours(sure_fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best_target = None
        min_distance = 9999.0 # 가장 가까운 거리 저장을 위한 초기값

        for c in cnts:
            area = cv2.contourArea(c)
            if 500 < area < 6000: # 낱개 사이즈 조건
                M = cv2.moments(c)
                if M["m00"] != 0:
                    u = int(M["m10"] / M["m00"]) + x1
                    v = int(M["m01"] / M["m00"]) + y1
                    
                    # [🔥 핵심] 해당 좌표의 거리(Depth) 측정
                    # mm 단위로 변환
                    dist = depth_f.get_distance(u, v) * 1000 
                    
                    # 0은 측정 오류이므로 제외, 이전 타겟보다 더 가까운(작은) 값이면 갱신
                    if 100 < dist < min_distance: 
                        min_distance = dist
                        
                        # 각도 계산 (minAreaRect)
                        rect = cv2.minAreaRect(c)
                        (cx, cy), (mw, mh), angle = rect
                        if mw < mh: angle += 90
                        
                        best_target = {
                            'u': u, 'v': v, 'dist': dist, 'angle': angle
                        }

        if best_target:
            u, v, dist, angle = best_target['u'], best_target['v'], best_target['dist'], best_target['angle']
            
            # 카메라 좌표 -> 로봇 좌표 변환
            intr = self.pipeline.get_active_profile().get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
            p_cam = rs.rs2_deproject_pixel_to_point(intr, [u, v], dist)
            dst = cv2.transform(np.array([[[p_cam[0], p_cam[1], p_cam[2]]]], dtype=np.float32), self.matrix)
            
            # 최종 좌표 설정 (보정값 포함)
            final_x = dst[0][0][0] + 15
            final_y = dst[0][0][1] - Y_OFFSET + 15
            final_z = dst[0][0][2] + Z_FINE_TUNE
            
            self.current_target = [final_x, final_y, final_z, FIXED_ORI[0], FIXED_ORI[1], angle]
            self.add_log(f"🎯 최단거리 타겟 확정! 거리: {dist:.1f}mm")
            self.add_log(f"📍 좌표: X={final_x:.1f}, Y={final_y:.1f}, Z={final_z:.1f}")
        else:
            self.current_target = None
            self.add_log("❌ 감지된 조각이 없거나 거리 측정 불가")

    def vision_worker(self):
        """실시간 화면 출력 (모든 감지된 후보를 보조적으로 표시)"""
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
                x1, x2 = w // 2 - 300, w // 2 + 100
                y1, y2 = h // 2 - 100, h - 100
                
                # 시각화용 마스크 처리
                crop = img[y1:y2, x1:x2]
                mask = cv2.inRange(cv2.cvtColor(crop, cv2.COLOR_BGR2HSV), LOWER_CHICKEN, UPPER_CHICKEN)
                dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
                _, fg = cv2.threshold(dist, 0.35 * dist.max() if dist.max()>0 else 0, 255, 0)
                fg = np.uint8(fg)
                cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                disp = img.copy()
                cv2.rectangle(disp, (x1, y1), (x2, y2), (255, 0, 0), 2)
                for c in cnts:
                    if cv2.contourArea(c) > 400:
                        cv2.drawContours(disp, [c + (x1, y1)], -1, (0, 255, 255), 2)
                
                self.root.after(0, self.update_cam_image, disp)
        finally:
            self.pipeline.stop()

    def send_command_and_wait(self, cmd):  
        if not self.conn: return False
        try:
            self.add_log(f"➡️ {cmd}")
            self.conn.setblocking(False)
            try:
                while self.conn.recv(1024): pass
            except: pass
            self.conn.setblocking(True)
            self.conn.sendall((cmd + "\n").encode())
            raw_data = self.conn.recv(1024).decode().strip()
            return "DONE" in raw_data
        except: return False

    def full_auto_loop(self):
        while self.auto_running:
            self.current_target = None
            self.add_log("🚚 안전 위치로 이동 중...")
            if not self.send_command_and_wait(f"MOVE,{SAFE_APPROACH[0]},{SAFE_APPROACH[1]},{SAFE_APPROACH[2]},0,180,0"): break
            
            time.sleep(1.5)
            self.scan_chicken()
            
            if self.current_target is None:
                self.add_log("💤 물체 없음... 재시도")
                time.sleep(1.0); continue

            tx, ty, tz, rx, ry, rzo = self.current_target
            fa = rzo - 90.0

            self.send_command_and_wait(f"MOVE,{tx:.2f},{ty:.2f},{SAFE_APPROACH[2]},0,180,0")
            self.send_command_and_wait(f"SET_J6,{fa:.2f}")
            if self.send_command_and_wait(f"DOWN,{tx:.2f},{ty:.2f},{tz:.2f},0,180,{fa:.2f}"):
                time.sleep(0.3)
                if self.send_command_and_wait("GRIP"):
                    time.sleep(1.0)
                    self.send_command_and_wait(f"MOVE,{tx:.2f},{ty:.2f},{SAFE_APPROACH[2]},0,180,{fa:.2f}")
            
            self.delivery_action()
            time.sleep(1)
            
        self.auto_running = False
        self.root.after(0, lambda: self.btn_auto_start.config(state=tk.NORMAL, bg="#f1c40f"))

    def send_command_and_wait(self, cmd):  
        if not self.conn: return False
        try:
            self.add_log(f"➡️ {cmd}")
            self.conn.setblocking(False)
            try:
                while self.conn.recv(1024): pass
            except: pass
            self.conn.setblocking(True)
            self.conn.sendall((cmd + "\n").encode())
            raw_data = self.conn.recv(1024).decode().strip()
            return "DONE" in raw_data
        except: return False

    def start_full_auto(self):
        if self.auto_running or not self.conn: return
        self.auto_running = True
        self.btn_auto_start.config(state=tk.DISABLED, bg="#7f8c8d")
        threading.Thread(target=self.full_auto_loop, daemon=True).start()

    def stop_full_auto(self): self.auto_running = False

    def get_robot_current_y(self):
        if not self.conn: return 0.0
        try:
            self.conn.sendall(b"GET\n"); resp = self.conn.recv(1024).decode().strip()
            return [float(x) for x in resp.split(',')][1] if resp else 0.0
        except: return 0.0
    
    def go_home(self):
        if self.get_robot_current_y() >= 190: self.go_safe()
        self.send_command(f"MOVE,{HOME_POS[0]},{HOME_POS[1]},{HOME_POS[2]},{HOME_POS[3]},{HOME_POS[4]},{HOME_POS[5]}")

    def go_safe(self): self.send_command(f"MOVE,{SAFE_APPROACH[0]},{SAFE_APPROACH[1]},{SAFE_APPROACH[2]},0,180,0")

    def go_target_top(self):
        if self.current_target: self.send_command(f"MOVE,{self.current_target[0]:.2f},{self.current_target[1]:.2f},{SAFE_APPROACH[2]},0,180,0")

    def down_and_grip(self):
        if self.current_target: threading.Thread(target=self._execute_down_grip, daemon=True).start()

    def _execute_down_grip(self):
        tx, ty, tz, _, _, rzo = self.current_target
        fa = rzo - 90.0
        self.send_command_and_wait(f"MOVE,{tx:.2f},{ty:.2f},{SAFE_APPROACH[2]},0,180,0")
        self.send_command_and_wait(f"SET_J6,{fa:.2f}")
        if self.send_command_and_wait(f"DOWN,{tx:.2f},{ty:.2f},{tz:.2f},0,180,{fa:.2f}"):
            time.sleep(0.3); self.send_command_and_wait("GRIP"); time.sleep(1.0)
            self.send_command_and_wait(f"MOVE,{tx:.2f},{ty:.2f},{SAFE_APPROACH[2]},0,180,{fa:.2f}")

    def delivery_action(self):
        self.send_command(f"MOVE,{HOME_POS[0]},{HOME_POS[1]},{HOME_POS[2]},{HOME_POS[3]},{HOME_POS[4]},{HOME_POS[5]}")
        self.send_command_and_wait("RELEASE")
        
    def update_cam_image(self, frame):
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).resize((640, 360))
        itk = ImageTk.PhotoImage(image=img)
        self.cam_label.itk = itk
        self.cam_label.configure(image=itk)
    
    def add_log(self, msg):
        self.log_widget.configure(state='normal')
        self.log_widget.insert(tk.END, f"[{time.strftime('%H:%M:%S')}] {msg}\n")
        self.log_widget.see(tk.END); self.log_widget.configure(state='disabled')
    
    def start_server(self):
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.bind(("0.0.0.0", PORT)); self.server.listen(1)
        threading.Thread(target=self.accept_conn, daemon=True).start()
    
    def accept_conn(self):
        while True: self.conn, _ = self.server.accept(); self.add_log("✅ 로봇 연결됨")
    
    def send_command(self, cmd):
        if self.conn: self.conn.sendall((cmd + "\n").encode()); return True
        return False

if __name__ == "__main__":
    root = tk.Tk(); app = ChickenMasterV4(root); root.mainloop()