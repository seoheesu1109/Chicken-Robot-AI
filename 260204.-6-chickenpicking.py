import socket
import threading
import time
import tkinter as tk
from tkinter import messagebox, scrolledtext
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
Y_OFFSET = -60.0  
MODEL_PATH = r"C:\runs\detect\train15\weights\best.pt"

class ChickenPickupMaster:
    def __init__(self, root):
        self.root = root
        self.conn = None
        self.is_connected = False
        self.current_target = None
        
        # YOLO 모델 로드
        try:
            self.model = YOLO(MODEL_PATH)
        except Exception as e:
            messagebox.showerror("Error", f"모델 로드 실패: {e}")

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
        self.root.title("Chicken Master (Center Crop 400x400)")
        self.root.geometry("680x950")
        
        self.log_widget = scrolledtext.ScrolledText(self.root, height=10, bg="#1E1E1E", fg="#00FF00")
        self.log_widget.pack(fill=tk.BOTH, padx=10, pady=5)

        btn_frame = tk.Frame(self.root)
        btn_frame.pack(fill=tk.BOTH, padx=20, pady=5)
        s = {"font": ("Arial", 10, "bold"), "height": 2}

        tk.Button(btn_frame, text="🏠 홈 이동", bg="#3498db", fg="white", command=self.go_home, **s).grid(row=0, column=0, sticky="nsew", padx=2, pady=2)
        tk.Button(btn_frame, text="🛡️ 안전위치", bg="#9b59b6", fg="white", command=self.go_safe, **s).grid(row=0, column=1, sticky="nsew", padx=2, pady=2)
        tk.Button(btn_frame, text="🔍 YOLO 감지 (중앙)", bg="#f1c40f", command=self.scan_chicken, **s).grid(row=1, column=0, sticky="nsew", padx=2, pady=2)
        tk.Button(btn_frame, text="📍 타겟상공 이동", bg="#e67e22", fg="white", command=self.go_target_top, **s).grid(row=1, column=1, sticky="nsew", padx=2, pady=2)
        tk.Button(btn_frame, text="🍗 하강 후 그립", bg="#e74c3c", fg="white", command=self.down_and_grip, **s).grid(row=2, column=0, sticky="nsew", padx=2, pady=2)
        tk.Button(btn_frame, text="🚚 배달 및 복귀", bg="#2ecc71", fg="white", command=self.delivery_path, **s).grid(row=2, column=1, sticky="nsew", padx=2, pady=2)
        
        for i in range(2): btn_frame.grid_columnconfigure(i, weight=1)

        self.cam_label = tk.Label(self.root, text="카메라 로딩 중...", bg="black")
        self.cam_label.pack(fill=tk.BOTH, padx=10, pady=10, expand=True)

    def get_crop_coords(self, w, h):
        """1280x720 기준 중앙 400x400 영역 계산"""
        x1, y1 = (w - 400) // 2, (h - 400) // 2
        x2, y2 = x1 + 400, y1 + 400
        return x1, y1, x2, y2

    def vision_worker(self):
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        profile = self.pipeline.start(config)
        self.align = rs.align(rs.stream.color)
        self.intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

        try:
            while True:
                frames = self.pipeline.wait_for_frames()
                aligned = self.align.process(frames)
                color_f = aligned.get_color_frame()
                if not color_f: continue
                
                img = np.asanyarray(color_f.get_data())
                h, w = img.shape[:2]
                x1, y1, x2, y2 = self.get_crop_coords(w, h)

                # 1. 크롭 및 YOLO 추론
                crop_img = img[y1:y2, x1:x2]
                results = self.model.predict(crop_img, conf=0.3, iou=0.45, verbose=False)
                
                # 2. 결과 시각화 (크롭된 이미지 위에 결과 그림)
                annotated_crop = results[0].plot()
                
                # 3. 전체 화면에 크롭 결과 덮어쓰기
                display_img = img.copy()
                display_img[y1:y2, x1:x2] = annotated_crop
                
                # 4. 크롭 영역 강조 (빨간 테두리)
                cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(display_img, "YOLO FOCUS AREA (400x400)", (x1, y1-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                self.root.after(0, self.update_cam_image, display_img)
        finally:
            self.pipeline.stop()

    def scan_chicken(self):
        self.add_log("📸 크롭 영역 집중 추론 시작...")
        frames = self.pipeline.wait_for_frames()
        aligned = self.align.process(frames)
        color_f = aligned.get_color_frame()
        depth_f = aligned.get_depth_frame()
        if not color_f or not depth_f: return

        img = np.asanyarray(color_f.get_data())
        h, w = img.shape[:2]
        x1, y1, x2, y2 = self.get_crop_coords(w, h)
        
        # 중앙 크롭 영역에서만 감지
        crop_img = img[y1:y2, x1:x2]
        results = self.model.predict(crop_img, conf=0.3, verbose=False)

        if len(results[0].boxes) > 0:
            box = results[0].boxes[0]
            xyxy = box.xyxy[0].cpu().numpy()
            
            # 크롭 이미지 기준 좌표 -> 전체 이미지 기준 좌표로 복원
            u = int((xyxy[0] + xyxy[2]) / 2) + x1
            v = int((xyxy[1] + xyxy[3]) / 2) + y1
            
            depth = depth_f.get_distance(u, v) * 1000
            if depth > 0:
                p_cam = rs.rs2_deproject_pixel_to_point(self.intrinsics, [u, v], depth)
                dst_pt = cv2.transform(np.array([[[p_cam[0], p_cam[1], p_cam[2]]]], dtype=np.float32), self.matrix)
                self.current_target = [dst_pt[0][0][0], dst_pt[0][0][1] + Y_OFFSET, dst_pt[0][0][2], 0, 180, 0]
                self.add_log(f"✅ 치킨 발견: X={self.current_target[0]:.1f}, Y={self.current_target[1]:.1f}")
        else:
            self.add_log("❌ 크롭 영역 내 감지된 치킨이 없습니다.")

    # --- [ 동작 로직 (기존과 동일) ] ---
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
            self.send_command(f"MOVE,{SAFE_APPROACH[0]},{SAFE_APPROACH[1]},{SAFE_APPROACH[2]},{SAFE_APPROACH[3]},{SAFE_APPROACH[4]},{SAFE_APPROACH[5]}")
        self.send_command(f"MOVE,{HOME_POS[0]},{HOME_POS[1]},{HOME_POS[2]},{HOME_POS[3]},{HOME_POS[4]},{HOME_POS[5]}")

    def go_safe(self):
        self.send_command(f"MOVE,{SAFE_APPROACH[0]},{SAFE_APPROACH[1]},{SAFE_APPROACH[2]},{SAFE_APPROACH[3]},{SAFE_APPROACH[4]},{SAFE_APPROACH[5]}")

    def go_target_top(self):
        if not self.current_target: return
        tx, ty, _, rx, ry, rz = self.current_target
        self.send_command(f"MOVE,{tx:.2f},{ty:.2f},{SAFE_APPROACH[2]},{rx},{ry},{rz}")

    def down_and_grip(self):
        if not self.current_target:
            self.add_log("⚠️ 타겟이 없습니다. 먼저 감지를 수행하세요.")
            return
            
        tx, ty, tz, rx, ry, rz = self.current_target
        
        # [수정] 하강 깊이(Z) 출력
        self.add_log(f"⬇️ 하강 시작 - 목표 깊이(Z): {tz:.2f}mm")
        
        # 1. 하강 명령
        # 만약 로봇이 너무 깊게 내려간다면 tz 대신 tz + 10 등으로 여유를 줄 수 있습니다.
        success = self.send_command(f"DOWN,{tx:.2f},{ty:.2f},{tz:.2f},{rx},{ry},{rz}")
        
        if success:
            time.sleep(0.3)
            # 2. 그리퍼 동작
            self.send_command("GRIP")
            time.sleep(0.5)
            # 3. 안전 상공으로 다시 상승
            self.send_command(f"MOVE,{tx:.2f},{ty:.2f},{SAFE_APPROACH[2]},{rx},{ry},{rz}")
            self.add_log(f"✅ 픽업 시도 완료 (Z: {tz:.2f})")
        else:
            self.add_log("❌ 하강 명령 전송 실패")

    def delivery_path(self):
        self.go_home()
        self.send_command("RELEASE")

    def update_cam_image(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        img = img.resize((600, 337), Image.LANCZOS)
        imgtk = ImageTk.PhotoImage(image=img)
        self.cam_label.imgtk = imgtk
        self.cam_label.configure(image=imgtk)

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
            self.add_log("✅ 로봇 연결 성공")
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

if __name__ == "__main__":
    root = tk.Tk()
    app = ChickenPickupMaster(root)
    root.mainloop()