import socket
import threading
import time
import math
import tkinter as tk
from tkinter import scrolledtext
from PIL import Image, ImageTk
import numpy as np
import cv2
import pyrealsense2 as rs
from ultralytics import YOLO

# [1. 설정 데이터]
PORT = 30002
FIXED_ORI = [0.0, 180.0, 0.0] 
HOME_POS = [424.17, -69.51, 742.91] + FIXED_ORI
SAFE_APPROACH = [728.90, 947.73, 742.91] + FIXED_ORI
SCAN_APPROACH = [728.90, 947.73, 372.51] + FIXED_ORI
Y_OFFSET = 60.0 
Z_FINE_TUNE = -70 # 100mm 더 하강하도록 설정

MODEL_PATH = r"C:\runs\detect\train15\weights\best.pt"

class ChickenMasterV4:
    def __init__(self, root):
        self.root = root
        self.conn = None
        self.pipeline = None
        self.current_target = None 
        
        self.setup_ui()
        
        try:
            self.model = YOLO(MODEL_PATH)
            self.add_log("✅ YOLO 모델 준비 완료")
        except: self.add_log("❌ 모델 로드 실패")

        # 캘리브레이션 데이터
        self.C_pts = np.array([[-60.99, -22.08, 751.00], [316.67, 13.45, 631.00], 
                               [-413.18, -43.74, 623.00], [-12.00, -276.04, 694.00], 
                               [-74.52, 253.81, 714.00]], dtype=np.float32)
        self.R_pts = np.array([[775.38, 893.70, 40.00], [410.45, 914.97, 178.26], 
                               [1142.18, 890.76, 174.17], [741.48, 641.75, 97.46], 
                               [789.80, 1178.78, 95.67]], dtype=np.float32)
        res = cv2.estimateAffine3D(self.C_pts, self.R_pts)
        self.matrix = res[1] if len(res) == 3 else res[0]

        threading.Thread(target=self.vision_worker, daemon=True).start()
        self.start_server()

    def setup_ui(self):
        self.root.title("Chicken Master V4 - Dynamic Orientation")
        self.root.geometry("950x980")
        self.root.configure(bg="#2c3e50")

        self.log_widget = scrolledtext.ScrolledText(self.root, height=10, bg="#1E1E1E", fg="#00FF00")
        self.log_widget.pack(fill=tk.X, padx=10, pady=5)

        btn_frame = tk.Frame(self.root, bg="#2c3e50")
        btn_frame.pack(fill=tk.X, padx=10, pady=5)
        
        s = {"font": ("맑은 고딕", 9, "bold"), "height": 2, "width": 14}

        tk.Button(btn_frame, text="🏠 홈 위치", bg="#34495e", fg="white", command=lambda: self.send_command(f"MOVE,{HOME_POS[0]},{HOME_POS[1]},{HOME_POS[2]},0,180,0"), **s).grid(row=1, column=0, padx=2, pady=2)
        tk.Button(btn_frame, text="📸 스캔 위치", bg="#34495e", fg="white", command=lambda: self.send_command(f"MOVE,{SCAN_APPROACH[0]},{SCAN_APPROACH[1]},{SCAN_APPROACH[2]},0,180,0"), **s).grid(row=1, column=1, padx=2, pady=2)
        tk.Button(btn_frame, text="🔍 치킨 스캔", bg="#f39c12", fg="white", command=self.scan_chicken, **s).grid(row=1, column=2, padx=2, pady=2)
        tk.Button(btn_frame, text="📍 타겟 상공", bg="#e67e22", fg="white", command=self.go_target_top, **s).grid(row=2, column=0, padx=2, pady=2)
        tk.Button(btn_frame, text="⚙️ J6 회전", bg="#d35400", fg="white", command=self.rotate_j6, **s).grid(row=2, column=1, padx=2, pady=2)
        tk.Button(btn_frame, text="👇 하강 피킹", bg="#e74c3c", fg="white", command=self.only_down_grip, **s).grid(row=2, column=2, padx=2, pady=2)
        tk.Button(btn_frame, text="🚚 배달", bg="#2ecc71", fg="white", command=self.delivery_action, **s).grid(row=3, column=0, padx=2, pady=2)
        tk.Button(btn_frame, text="👐 오픈", bg="#95a5a6", fg="white", command=lambda: self.send_command("RELEASE"), **s).grid(row=3, column=1, padx=2, pady=2)

        self.cam_label = tk.Label(self.root, bg="black")
        self.cam_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    def get_chicken_orientation(self, roi):
        """치킨의 대각선 방향을 정확히 잡기 위한 개선된 로직"""
        # 1. 컬러 기반 추출 (치킨의 주황/갈색 계열만 강조)
        # 흑백 변환 대신 채도를 높여 치킨 형체만 뚜렷하게 만듭니다.
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        # 치킨 색상 범위 (이 부분을 조절하면 바닥 노이즈를 무시합니다)
        lower_orange = np.array([5, 50, 50])
        upper_orange = np.array([25, 255, 255])
        mask = cv2.inRange(hsv, lower_orange, upper_orange)
        
        # 2. 노이즈 제거 (가우시안 블러 및 모폴로지)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel) # 잔상 제거
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel) # 구멍 메우기
        
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts: return None
        
        # 3. 가장 큰 영역(치킨) 선택
        cnt = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(cnt) < 300: return None
        
        # 4. 회전 사각형 계산
        rect = cv2.minAreaRect(cnt)
        (cx_rel, cy_rel), (w, h), angle = rect
        
        # 5. [중요] OpenCV 각도 체계 보정
        # 장축(긴 쪽)이 어디냐에 따라 정확한 대각선 각도를 산출합니다.
        if w < h:
            angle = angle + 180 if angle < 0 else angle # 0~180도 유지
        else:
            angle = angle + 90
            
        return cx_rel, cy_rel, angle, max(w, h)

    def vision_worker(self):
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        cfg.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        self.pipeline.start(cfg)
        
        try:
            while True:
                f = self.pipeline.wait_for_frames()
                color_f = f.get_color_frame()
                if not color_f: continue
                
                img = np.asanyarray(color_f.get_data())
                res = self.model.predict(img, conf=0.15, verbose=False)
                disp = img.copy()

                for r in res:
                    for box in r.boxes:
                        b = box.xyxy[0].cpu().numpy().astype(int)
                        x1, y1, x2, y2 = b
                        roi = img[y1:y2, x1:x2]
                        if roi.size == 0: continue
                        
                        orient = self.get_chicken_orientation(roi)
                        if orient:
                            cx_rel, cy_rel, angle, length = orient
                            cx, cy = int(cx_rel + x1), int(cy_rel + y1)
                            
                            # 치킨 길이에 맞춘 동적 선 그리기 (중심에서 양방향)
                            half_l = length / 2
                            rad = math.radians(angle)
                            rx = int(cx + half_l * math.cos(rad))
                            ry = int(cy + half_l * math.sin(rad))
                            lx = int(cx - half_l * math.cos(rad))
                            ly = int(cy - half_l * math.sin(rad))
                            
                            # 시각화: 장축(빨강), 중심점(초록)
                            cv2.line(disp, (lx, ly), (rx, ry), (0, 0, 255), 3) 
                            cv2.circle(disp, (cx, cy), 6, (0, 255, 0), -1)
                            cv2.putText(disp, f"{angle:.1f}deg", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                self.root.after(0, self.update_cam_image, disp)
        finally: self.pipeline.stop()

    def scan_chicken(self):
        self.add_log("🔍 정밀 스캔 실행...")
        f = self.pipeline.wait_for_frames()
        a = rs.align(rs.stream.color).process(f)
        color_f, depth_f = a.get_color_frame(), a.get_depth_frame()
        if not color_f or not depth_f: return
        
        img = np.asanyarray(color_f.get_data())
        results = self.model.predict(img, conf=0.15, verbose=False)
        
        valid_targets = []
        for r in results:
            for box in r.boxes:
                b = box.xyxy[0].cpu().numpy().astype(int)
                roi = img[b[1]:b[3], b[0]:b[2]]
                orient = self.get_chicken_orientation(roi)
                
                if orient:
                    cx_rel, cy_rel, angle, _ = orient
                    u, v = int(cx_rel + b[0]), int(cy_rel + b[1])
                    dist = depth_f.get_distance(u, v) * 1000 
                    if dist > 0:
                        valid_targets.append({'u': u, 'v': v, 'dist': dist, 'angle': angle})

        if valid_targets:
            target = min(valid_targets, key=lambda t: t['v'])
            intr = self.pipeline.get_active_profile().get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
            p_cam = rs.rs2_deproject_pixel_to_point(intr, [target['u'], target['v']], target['dist'])
            dst = cv2.transform(np.array([[[p_cam[0], p_cam[1], p_cam[2]]]], dtype=np.float32), self.matrix)
            
            # 높이 보정
            height_diff = SAFE_APPROACH[2] - SCAN_APPROACH[2]
            final_x = dst[0][0][0] - 40
            final_y = dst[0][0][1] - Y_OFFSET + 5
            final_z = (dst[0][0][2] - height_diff) + Z_FINE_TUNE 
            
            self.current_target = [final_x, final_y, final_z, FIXED_ORI[0], FIXED_ORI[1], target['angle']]
            self.add_log(f"🎯 타겟 확정: X{final_x:.1f} Y{final_y:.1f} Z{final_z:.1f} / 각도 {target['angle']:.1f}")
        else: self.add_log("❌ 치킨을 찾을 수 없습니다.")

    # (이하 함수들은 이전과 동일)
    def go_target_top(self):
        if not self.current_target: return
        tx, ty = self.current_target[0], self.current_target[1]
        self.send_command(f"MOVE,{tx:.2f},{ty:.2f},{SCAN_APPROACH[2]},0,180,0")

    def rotate_j6(self):
        if not self.current_target: return
        fa = self.current_target[5] - 0
        self.send_command(f"SET_J6,{fa:.2f}")

    def only_down_grip(self):
        if not self.current_target: return
        tx, ty, tz, _, _, ang = self.current_target
        fa = ang - -30.0 # 하강 6축 보정
        threading.Thread(target=self._exec_pick, args=(tx, ty, tz, fa), daemon=True).start()

    def _exec_pick(self, tx, ty, tz, fa):
        self.send_command_and_wait(f"DOWN,{tx:.2f},{ty:.2f},{tz:.2f},0,180,{fa:.2f}")
        time.sleep(0.2)
        self.send_command_and_wait("GRIP")
        time.sleep(0.2)
        self.send_command_and_wait(f"MOVE,{tx:.2f},{ty:.2f},{SCAN_APPROACH[2]},0,180,{fa:.2f}")

    def update_cam_image(self, frame):
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).resize((800, 450))
        itk = ImageTk.PhotoImage(image=img); self.cam_label.itk = itk; self.cam_label.configure(image=itk)

    def add_log(self, msg):
        self.log_widget.configure(state='normal'); self.log_widget.insert(tk.END, f"[{time.strftime('%H:%M:%S')}] {msg}\n"); self.log_widget.see(tk.END); self.log_widget.configure(state='disabled')

    def send_command(self, cmd):
        if self.conn: self.conn.sendall((cmd + "\n").encode()); self.add_log(f"➡️ {cmd}")

    def send_command_and_wait(self, cmd):
        if not self.conn: return False
        self.add_log(f"➡️ {cmd}")
        self.conn.setblocking(False)
        try:
            while self.conn.recv(1024): pass
        except: pass
        self.conn.setblocking(True)
        self.conn.sendall((cmd + "\n").encode())
        res = self.conn.recv(1024).decode()
        return "DONE" in res

    def delivery_action(self):
        self.send_command_and_wait(f"MOVE,{HOME_POS[0]},{HOME_POS[1]},{HOME_POS[2]},0,180,0")
        self.send_command("RELEASE")

    def start_server(self):
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM); self.server.bind(("0.0.0.0", PORT)); self.server.listen(1)
        threading.Thread(target=self.accept_conn, daemon=True).start()

    def accept_conn(self):
        while True: self.conn, _ = self.server.accept(); self.add_log("✅ 로봇 연결 성공")

if __name__ == "__main__":
    root = tk.Tk(); app = ChickenMasterV4(root); root.mainloop()