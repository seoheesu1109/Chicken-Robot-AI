import socket
import threading
import time
import tkinter as tk
from tkinter import scrolledtext
from PIL import Image, ImageTk
import numpy as np
import cv2
import pyrealsense2 as rs
import requests

# [1. 설정 데이터]
PORT = 30002
FIXED_ORI = [0.0, 180.0, 0.0] 
HOME_POS = [424.17, -69.51, 742.91] + FIXED_ORI
SAFE_APPROACH = [728.90, 947.73, 742.91] + FIXED_ORI
Y_OFFSET = 60.0 
# 고정 Z_OFFSET은 이제 사용하지 않거나, 필요 시 미세 조정용으로만 씁니다.
Z_FINE_TUNE = 30

# [HSV 범위 - 치킨/물체 색상]
LOWER_CHICKEN = np.array([10, 70, 110])    
UPPER_CHICKEN = np.array([28, 255, 220]) 

class ChickenMasterV4:
    def __init__(self, root):
        self.root = root
        self.conn = None
        self.pipeline = None
        self.current_target = None # [x, y, z, rx, ry, rz]
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
        self.root.title("Robot Control Center - Real-time Z Detection")
        self.root.geometry("800x980")
        self.root.configure(bg="#2c3e50")

        self.log_widget = scrolledtext.ScrolledText(self.root, height=8, bg="#1E1E1E", fg="#00FF00")
        self.log_widget.pack(fill=tk.X, padx=10, pady=5)

        main_btn_frame = tk.Frame(self.root, bg="#2c3e50")
        main_btn_frame.pack(fill=tk.X, padx=20, pady=5)
        
        s = {"font": ("맑은 고딕", 10, "bold"), "height": 2, "width": 15}

        tk.Button(main_btn_frame, text="🏠 홈 이동", bg="#3498db", fg="white", command=self.go_home, **s).grid(row=0, column=0, padx=2, pady=2)
        tk.Button(main_btn_frame, text="🛡️ 안전위치", bg="#9b59b6", fg="white", command=self.go_safe, **s).grid(row=0, column=1, padx=2, pady=2)
        tk.Button(main_btn_frame, text="🔍 치킨 감지", bg="#f39c12", fg="white", command=self.scan_chicken, **s).grid(row=0, column=2, padx=2, pady=2)

        tk.Button(main_btn_frame, text="🍗 하강 후 그립", bg="#e74c3c", fg="white", command=self.down_and_grip, **s).grid(row=1, column=0, padx=2, pady=2)
        tk.Button(main_btn_frame, text="🚚 배출(배달)", bg="#2ecc71", fg="white", command=self.delivery_action, **s).grid(row=1, column=1, padx=2, pady=2)
        tk.Button(main_btn_frame, text="📍 타겟상공", bg="#e67e22", fg="white", command=self.go_target_top, **s).grid(row=1, column=2, padx=2, pady=2)

        tk.Button(main_btn_frame, text="✊ GRIP", bg="#27ae60", fg="white", command=lambda: self.send_command("GRIP"), **s).grid(row=2, column=0, padx=2, pady=2)
        tk.Button(main_btn_frame, text="✋ RELEASE", bg="#95a5a6", fg="white", command=lambda: self.send_command("RELEASE"), **s).grid(row=2, column=1, padx=2, pady=2)

        self.btn_auto_start = tk.Button(main_btn_frame, text="🤖 풀오토 시작", bg="#f1c40f", fg="black", command=self.start_full_auto, **s)
        self.btn_auto_start.grid(row=3, column=0, padx=2, pady=5)
        
        self.btn_auto_stop = tk.Button(main_btn_frame, text="🛑 무한루프 중단", bg="#e67e22", fg="white", command=self.stop_full_auto, **s)
        self.btn_auto_stop.grid(row=3, column=1, padx=2, pady=5)

        self.cam_label = tk.Label(self.root, bg="black"); self.cam_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    # --- 실시간 Z값 추출이 포함된 핵심 함수 ---
    def scan_chicken(self):
        self.add_log("🔍 최상단 타겟 스캔 및 좌표 보정 중...")
        f = self.pipeline.wait_for_frames()
        a = rs.align(rs.stream.color).process(f)
        color_f, depth_f = a.get_color_frame(), a.get_depth_frame()
        if not color_f or not depth_f: return
        
        img = np.asanyarray(color_f.get_data())
        h, w = img.shape[:2]
        
        # 크롭 영역 설정 (화면 하단 중앙부)
        x1 = w // 2 - 300  # 가로폭을 좌측으로 더 확장 (250 -> 350)
        x2 = w // 2 + 100  # 가로폭을 우측으로 더 확장 (250 -> 350)
        y1 = h // 2 - 100  # 원의 상단부터 시작 (전체 높이의 중간 지점 부근)
        y2 = h - 100        # 바닥에서 약간 띄움
        mask = cv2.inRange(cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2HSV), LOWER_CHICKEN, UPPER_CHICKEN)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        valid_targets = []
        for c in cnts:
            if cv2.contourArea(c) > 800:
                rect = cv2.minAreaRect(c)
                (cx_crop, cy_crop), (mw, mh), angle = rect
                # 픽셀 좌표 복원
                u, v = int(cx_crop) + x1, int(cy_crop) + y1
                valid_targets.append({
                    'u': u, 'v': v, 'angle': angle, 'mw': mw, 'mh': mh
                })

        if valid_targets:
            # 1. 화면상 가장 위쪽(Y 픽셀 좌표 v가 가장 작은 것) 선택
            top_target = min(valid_targets, key=lambda t: t['v'])
            
            u, v = top_target['u'], top_target['v']
            angle = top_target['angle']
            if top_target['mw'] < top_target['mh']: angle += 90
            
            # 2. 실시간 깊이 측정 (Z값)
            dist = depth_f.get_distance(u, v) * 1000 # mm 단위
            
            if dist > 0:
                # 카메라 좌표계를 로봇 좌표계로 변환
                intr = self.pipeline.get_active_profile().get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
                p_cam = rs.rs2_deproject_pixel_to_point(intr, [u, v], dist)
                dst = cv2.transform(np.array([[[p_cam[0], p_cam[1], p_cam[2]]]], dtype=np.float32), self.matrix)
                
                # --- [수정 포인트: 강제 보정 적용] ---
                # 로봇 좌표계 기준 변환값에 각각 20mm씩 더함
                final_x = dst[0][0][0] + 15
                final_y = dst[0][0][1] - Y_OFFSET + 15
                final_z = dst[0][0][2] + Z_FINE_TUNE  # 실시간 측정 Z값 사용
                
                # 타겟 데이터 업데이트
                self.current_target = [final_x, final_y, final_z, FIXED_ORI[0], FIXED_ORI[1], angle]
                
                self.add_log(f"🎯 최상단 타겟 고정 (+20mm 보정)")
                self.add_log(f"📍 좌표: X={final_x:.1f}, Y={final_y:.1f}, Z={final_z:.1f}")
            else:
                self.add_log("⚠️ 깊이 측정 실패 (0.0m)")
        else:
            self.current_target = None
            self.add_log("❌ 감지된 물체 없음")

    def full_auto_loop(self):
        while self.auto_running:
            # 1. 스캔 위치로 이동
            if not self.send_command_and_wait(f"MOVE,{SAFE_APPROACH[0]},{SAFE_APPROACH[1]},{SAFE_APPROACH[2]},0,180,0"): break
            time.sleep(0.5)

            # 2. 감지 (여기서 실시간 Z값이 current_target에 저장됨)
            self.scan_chicken()
            
            if self.current_target is None:
                self.add_log("💤 물체 없음... 재시도")
                time.sleep(1.5)
                continue

            # 3. 집기 시퀀스
            tx, ty, tz, rx, ry, rz_origin = self.current_target
            final_angle = rz_origin - 75.0

            self.send_command_and_wait(f"MOVE,{tx:.2f},{ty:.2f},{SAFE_APPROACH[2]},0,180,0")
            self.send_command_and_wait(f"SET_J6,{final_angle:.2f}")
            # 여기서 tz는 실시간으로 측정된 깊이값입니다.
            self.send_command_and_wait(f"DOWN,{tx:.2f},{ty:.2f},{tz:.2f},0,180,{final_angle:.2f}")
            self.send_command_and_wait("GRIP")
            self.send_command_and_wait(f"MOVE,{tx:.2f},{ty:.2f},{SAFE_APPROACH[2]},0,180,{final_angle:.2f}")

            # 4. 배출
            self.send_command_and_wait(f"MOVE,{HOME_POS[0]},{HOME_POS[1]},{HOME_POS[2]},0,180,0")
            self.send_command_and_wait("RELEASE")
            self.add_log("✅ 사이클 완료")
            self.send_discord_alert("🍗 치킨 배달 완료! 맛나게 드십쇼.")
            time.sleep(0.5)

        self.go_home()
        self.auto_running = False
        self.root.after(0, lambda: self.btn_auto_start.config(state=tk.NORMAL, bg="#f1c40f"))

    # --- 기존 나머지 함수 유지 ---
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
                x1 = w // 2 - 300  # 가로폭을 좌측으로 더 확장 (250 -> 350)
                x2 = w // 2 + 100  # 가로폭을 우측으로 더 확장 (250 -> 350)
                y1 = h // 2 - 100  # 원의 상단부터 시작 (전체 높이의 중간 지점 부근)
                y2 = h - 100        # 바닥에서 약간 띄움
                crop = img[y1:y2, x1:x2]
                mask = cv2.inRange(cv2.cvtColor(crop, cv2.COLOR_BGR2HSV), LOWER_CHICKEN, UPPER_CHICKEN)
                cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                disp = img.copy()
                cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 255), 2)
                for c in cnts:
                    if cv2.contourArea(c) > 800:
                        cv2.drawContours(disp, [c + (x1, y1)], -1, (0, 255, 0), 2)
                self.root.after(0, self.update_cam_image, disp)
        finally: self.pipeline.stop()

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
        if self.auto_running: return
        if not self.conn: self.add_log("⚠️ 연결 확인!"); return
        self.auto_running = True
        self.btn_auto_start.config(state=tk.DISABLED, bg="#7f8c8d")
        threading.Thread(target=self.full_auto_loop, daemon=True).start()
    def stop_full_auto(self): self.auto_running = False
    def get_robot_current_y(self):
        if not self.conn: return 0.0
        try:
            self.conn.sendall(b"GET\n")
            resp = self.conn.recv(1024).decode().strip()
            if resp:
                coords = [float(x) for x in resp.split(',')]
                return coords[1]
        except: return 0.0
    def go_home(self):
        cy = self.get_robot_current_y()
        if cy >= 190:
            self.send_command(f"MOVE,{SAFE_APPROACH[0]},{SAFE_APPROACH[1]},{SAFE_APPROACH[2]},0,180,0")
            time.sleep(0.2)
        self.send_command(f"MOVE,{HOME_POS[0]},{HOME_POS[1]},{HOME_POS[2]},{HOME_POS[3]},{HOME_POS[4]},{HOME_POS[5]}")
    def go_safe(self): self.send_command(f"MOVE,{SAFE_APPROACH[0]},{SAFE_APPROACH[1]},{SAFE_APPROACH[2]},0,180,0")
    def go_target_top(self):
        if self.current_target:
            self.send_command(f"MOVE,{self.current_target[0]:.2f},{self.current_target[1]:.2f},{SAFE_APPROACH[2]},0,180,0")

    def down_and_grip(self):
        if self.current_target: threading.Thread(target=self._execute_down_grip, daemon=True).start()

    def _execute_down_grip(self):
        tx, ty, tz, rx, ry, rzo = self.current_target
        fa = rzo - 90.0
        
        # 1. 타겟 상공 이동
        self.send_command_and_wait(f"MOVE,{tx:.2f},{ty:.2f},{SAFE_APPROACH[2]},0,180,0")
        # 2. J6 회전
        self.send_command_and_wait(f"SET_J6,{fa:.2f}")
        
        # 3. 힘 감지 하강 (DOWN) - DONE이 올 때까지 대기
        self.add_log("⬇️ 하강 시작 (힘 감지 대기...)")
        if self.send_command_and_wait(f"DOWN,{tx:.2f},{ty:.2f},{tz:.2f},0,180,{fa:.2f}"):
            # [핵심] 로봇이 멈추고 제어기가 IDLE 상태로 복귀할 아주 짧은 시간 부여
            time.sleep(0.3) 
            
            # 4. 그립 실행
            self.add_log("✊ 접촉 확인, 그립 실행")
            if self.send_command_and_wait("GRIP"):
                time.sleep(0.5) # 그립 완료 대기
                # 5. 복귀
                self.send_command_and_wait(f"MOVE,{tx:.2f},{ty:.2f},{SAFE_APPROACH[2]},0,180,{fa:.2f}")
        else:
            self.add_log("❌ 하강 중 에러 발생")
    def delivery_action(self):
        """배출(배달) 버튼을 눌렀을 때 실행될 동작"""
        self.add_log("🚚 배출 동작을 시작합니다.")
        
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
        if self.conn: self.conn.sendall((cmd + "\n").encode()); return True
        return False

if __name__ == "__main__":
    root = tk.Tk(); app = ChickenMasterV4(root); root.mainloop()