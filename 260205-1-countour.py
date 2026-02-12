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
SAFE_APPROACH = [728.90, 947.73, 742.91] # 스캔을 위한 상공 위치
SAFE_Z = 742.91                          # 이동 및 복귀 높이
PICK_Z = 35.0                            # 하강 목표 높이 (바닥 기준)
Y_OFFSET = -60.0                         # 카메라-그리퍼 간격 보정

# [HSV 범위] - 노란색 치킨 감지
LOWER_CHICKEN = np.array([10, 70, 110])    
UPPER_CHICKEN = np.array([28, 255, 220]) 

class ChickenMasterV7:
    def __init__(self, root):
        self.root = root
        self.conn = None
        self.pipeline = None
        self.current_target = None # [x, y, z, angle]
        
        # 캘리브레이션 행렬 (카메라 -> 로봇 좌표 변환)
        self.matrix = self.init_calibration()

        self.setup_ui()
        
        # 비전 및 서버 스레드 시작
        threading.Thread(target=self.vision_worker, daemon=True).start()
        threading.Thread(target=self.start_server, daemon=True).start()

    def init_calibration(self):
        C_pts = np.array([[-60.99, -22.08, 751.00], [316.67, 13.45, 631.00], 
                          [-413.18, -43.74, 623.00], [-12.00, -276.04, 694.00], 
                          [-74.52, 253.81, 714.00]], dtype=np.float32)
        R_pts = np.array([[775.38, 893.70, 40.00], [410.45, 914.97, 178.26], 
                          [1142.18, 890.76, 174.17], [741.48, 641.75, 97.46], 
                          [789.80, 1178.78, 95.67]], dtype=np.float32)
        res = cv2.estimateAffine3D(C_pts, R_pts)
        return res[1] if len(res) == 3 else res[0]

    def setup_ui(self):
        self.root.title("Chicken Automation V7 - J6 Angle Correction")
        self.root.geometry("850x950")
        self.root.configure(bg="#2c3e50")

        # 로그창
        self.log_widget = scrolledtext.ScrolledText(self.root, height=12, bg="#1E1E1E", fg="#00FF00", font=("Consolas", 10))
        self.log_widget.pack(fill=tk.X, padx=10, pady=5)

        # 버튼 영역
        btn_frame = tk.Frame(self.root, bg="#2c3e50")
        btn_frame.pack(fill=tk.X, padx=10, pady=5)
        
        btn_style = {"width": 25, "height": 2, "font": ("맑은 고딕", 10, "bold")}
        tk.Button(btn_frame, text="🔍 1. 치킨 감지 (스캔 위치로)", command=self.scan_chicken, bg="#f39c12", fg="white", **btn_style).pack(side=tk.LEFT, padx=10)
        tk.Button(btn_frame, text="🍗 2. 자동 잡기 및 들어올리기", command=self.start_pick_process, bg="#e74c3c", fg="white", **btn_style).pack(side=tk.LEFT, padx=10)
        
        # 카메라 화면
        self.cam_label = tk.Label(self.root, bg="black")
        self.cam_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    def add_log(self, msg):
        self.log_widget.configure(state='normal')
        self.log_widget.insert(tk.END, f"[{time.strftime('%H:%M:%S')}] {msg}\n")
        self.log_widget.see(tk.END)
        self.log_widget.configure(state='disabled')

    def vision_worker(self):
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        cfg.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        self.pipeline.start(cfg)
        align = rs.align(rs.stream.color)
        
        try:
            while True:
                frames = self.pipeline.wait_for_frames()
                aligned = align.process(frames)
                color_f = aligned.get_color_frame()
                if not color_f: continue
                
                img = np.asanyarray(color_f.get_data())
                h, w = img.shape[:2]
                x1, y1, x2, y2 = (w//2 - 250), (h - 500), (w//2 + 250), h
                
                # 화면 가이드라인 및 인식 박스 시각화
                disp = img.copy()
                cv2.rectangle(disp, (x1, y1), (x2, y2), (255, 0, 0), 2)
                
                # UI 업데이트 (640x360으로 리사이즈하여 출력)
                self.update_ui_image(disp)
        finally:
            self.pipeline.stop()

    def update_ui_image(self, frame):
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).resize((720, 405))
        itk = ImageTk.PhotoImage(image=img)
        self.cam_label.itk = itk
        self.cam_label.configure(image=itk)

    def send_wait(self, cmd):
        if not self.conn: return False
        try:
            self.add_log(f"➡️ 전송: {cmd}")
            self.conn.sendall((cmd + "\n").encode())
            resp = self.conn.recv(1024).decode().strip()
            if "DONE" in resp: return True
        except: pass
        return False

    def scan_chicken(self):
        if not self.conn:
            self.add_log("⚠️ 로봇 연결을 확인하세요.")
            return

        # 1. 지정된 SAFE_APPROACH 위치로 이동
        self.add_log(f"🚀 스캔 위치로 이동: {SAFE_APPROACH[:3]}")
        move_cmd = f"MOVE,{SAFE_APPROACH[0]},{SAFE_APPROACH[1]},{SAFE_APPROACH[2]},0.0,180.0,0.0"
        if not self.send_wait(move_cmd): return

        time.sleep(0.5) # 진동 방지 대기

        # 2. 치킨 스캔 및 좌표 계산
        self.add_log("🔍 이미지 분석 중...")
        frames = self.pipeline.wait_for_frames()
        aligned = rs.align(rs.stream.color).process(frames)
        color_f = aligned.get_color_frame()
        depth_f = aligned.get_depth_frame()
        
        img = np.asanyarray(color_f.get_data())
        h, w = img.shape[:2]
        x1, y1, x2, y2 = (w//2 - 250), (h - 500), (w//2 + 250), h
        crop = img[y1:y2, x1:x2]
        
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, LOWER_CHICKEN, UPPER_CHICKEN)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        valid = [c for c in cnts if cv2.contourArea(c) > 1000]
        if valid:
            c = max(valid, key=cv2.contourArea)
            rect = cv2.minAreaRect(c)
            (cx, cy), (mw, mh), angle = rect
            if mw < mh: angle += 90 # 장축 정렬
            
            u, v = int(cx) + x1, int(cy) + y1
            dist = depth_f.get_distance(u, v) * 1000
            
            intr = self.pipeline.get_active_profile().get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
            p_cam = rs.rs2_deproject_pixel_to_point(intr, [u, v], dist)
            p_robot = cv2.transform(np.array([[[p_cam[0], p_cam[1], p_cam[2]]]], dtype=np.float32), self.matrix)
            
            # 타겟 저장 (X, Y_OFFSET 적용 Y, 기본 Z, 원본 각도)
            self.current_target = [p_robot[0][0][0], p_robot[0][0][1] + Y_OFFSET, p_robot[0][0][2], angle]
            self.add_log(f"✅ 감지 성공! X:{self.current_target[0]:.1f}, 원본 각도:{angle:.1f}")
        else:
            self.add_log("❌ 치킨을 찾지 못했습니다.")

    def start_pick_process(self):
        if not self.current_target:
            self.add_log("⚠️ 스캔을 먼저 실행하세요.")
            return
        threading.Thread(target=self.execute_pick, daemon=True).start()

    def execute_pick(self):
        tx, ty, tz, angle = self.current_target
        
        # [수정] 강제로 -90도 보정 적용
        final_angle = angle - 90.0
        
        # 1. 타겟 상공 이동
        if not self.send_wait(f"MOVE,{tx:.2f},{ty:.2f},{SAFE_Z},0.0,180.0,0.0"): return
        time.sleep(0.2)
        
        # 2. 6축(J6)만 회전 (강제 보정 각도로)
        if not self.send_wait(f"SET_J6,{final_angle:.2f}"): return
        time.sleep(0.2)
        
        # 3. 하강 (보정된 각도 유지)
        if not self.send_wait(f"DOWN,{tx:.2f},{ty:.2f},{PICK_Z:.2f},0.0,180.0,{final_angle:.2f}"): return
        time.sleep(0.2)
        
        # 4. 잡기 (GRIP)
        if not self.send_wait("GRIP"): return
        time.sleep(0.2)
        
        # 5. 복귀 (들어올리기)
        if not self.send_wait(f"MOVE,{tx:.2f},{ty:.2f},{SAFE_Z},0.0,180.0,{final_angle:.2f}"): return
        time.sleep(0.2)
        
        self.add_log(f"🎉 성공! 보정 각도 {final_angle:.1f}도로 집어올렸습니다.")

    def start_server(self):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.bind(("0.0.0.0", PORT))
        server.listen(1)
        while True:
            self.conn, _ = server.accept()
            self.add_log("🤖 로봇 통신 서버 활성화 (로봇 연결됨)")

if __name__ == "__main__":
    root = tk.Tk()
    app = ChickenMasterV7(root)
    root.mainloop()