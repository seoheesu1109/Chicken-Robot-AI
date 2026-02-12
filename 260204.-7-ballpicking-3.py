import socket
import threading
import time
import tkinter as tk
from tkinter import messagebox, scrolledtext
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

class BallPickupDetailControl:
    def __init__(self, root):
        self.root = root
        self.conn = None
        self.is_connected = False
        self.current_target = None 
        
        # 캘리브레이션 데이터
        self.C_pts = np.array([[-60.99, -22.08, 751.00], [316.67, 13.45, 631.00], [-413.18, -43.74, 623.00], [-12.00, -276.04, 694.00], [-74.52, 253.81, 714.00]], dtype=np.float32)
        self.R_pts = np.array([[775.38, 893.70, 40.00], [410.45, 914.97, 178.26], [1142.18, 890.76, 174.17], [741.48, 641.75, 97.46], [789.80, 1178.78, 95.67]], dtype=np.float32)
        res = cv2.estimateAffine3D(self.C_pts, self.R_pts)
        self.matrix = res[1] if len(res) == 3 else res[0]

        self.setup_ui()
        
        # 비전 및 서버 스레드 시작
        threading.Thread(target=self.vision_worker, daemon=True).start()
        self.start_server()

    def setup_ui(self):
        self.root.title("Ball Robot Integrated Control (Hough Circles)")
        self.root.geometry("640x950")
        
        self.log_widget = scrolledtext.ScrolledText(self.root, height=10, bg="#f0f0f0")
        self.log_widget.pack(fill=tk.BOTH, padx=10, pady=5)

        btn_frame = tk.Frame(self.root)
        btn_frame.pack(fill=tk.BOTH, padx=20, pady=5)
        s = {"font": ("Arial", 10, "bold"), "height": 2}

        tk.Button(btn_frame, text="🏠 홈 이동", bg="#3498db", fg="white", command=self.go_home, **s).grid(row=0, column=0, sticky="nsew", padx=2, pady=2)
        tk.Button(btn_frame, text="🛡️ 안전위치", bg="#9b59b6", fg="white", command=self.go_safe, **s).grid(row=0, column=1, sticky="nsew", padx=2, pady=2)
        tk.Button(btn_frame, text="🔍 XY 보정 (인식)", bg="#f1c40f", command=self.scan_target, **s).grid(row=1, column=0, sticky="nsew", padx=2, pady=2)
        tk.Button(btn_frame, text="📍 타겟상공 이동", bg="#e67e22", fg="white", command=self.go_target_top, **s).grid(row=1, column=1, sticky="nsew", padx=2, pady=2)
        tk.Button(btn_frame, text="👇 하강 후 그립", bg="#e74c3c", fg="white", command=self.down_and_grip, **s).grid(row=2, column=0, sticky="nsew", padx=2, pady=2)
        tk.Button(btn_frame, text="🚚 배송 (안전경유)", bg="#2ecc71", fg="white", command=self.delivery_path, **s).grid(row=2, column=1, sticky="nsew", padx=2, pady=2)
        
        for i in range(2): btn_frame.grid_columnconfigure(i, weight=1)

        self.cam_label = tk.Label(self.root, text="카메라 연결 중...", bg="black")
        self.cam_label.pack(fill=tk.BOTH, padx=10, pady=10, expand=True)

    def update_cam_image(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        img = img.resize((600, 337), Image.LANCZOS)
        imgtk = ImageTk.PhotoImage(image=img)
        self.cam_label.imgtk = imgtk
        self.cam_label.configure(image=imgtk)

    def vision_worker(self):
        """실시간 화면 출력 및 모든 원 검출 시각화"""
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        profile = self.pipeline.start(config)
        self.align = rs.align(rs.stream.color)
        
        try:
            while True:
                frames = self.pipeline.wait_for_frames()
                aligned = self.align.process(frames)
                color_f = aligned.get_color_frame()
                if not color_f: continue
                
                img = np.asanyarray(color_f.get_data())
                display_img = img.copy()
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                # HSV 가우시안 블러 및 파란색 필터링
                hsv = cv2.cvtColor(cv2.GaussianBlur(img, (11, 11), 0), cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv, np.array([85, 100, 100]), np.array([105, 255, 255]))
                
                # 마스크 영역만 추출하여 원 검출 효율 극대화
                masked_gray = cv2.bitwise_and(gray, gray, mask=mask)

                # 허프 원 변환 (Hough Circles)
                circles = cv2.HoughCircles(
                    masked_gray, 
                    cv2.HOUGH_GRADIENT, dp=1.2, minDist=40, 
                    param1=50, param2=25, minRadius=20, maxRadius=60
                )

                if circles is not None:
                    circles = np.uint16(np.around(circles))
                    for i, (u, v, r) in enumerate(circles[0]):
                        # 모든 인식된 원 그리기
                        cv2.circle(display_img, (u, v), r, (0, 255, 0), 2)
                        cv2.circle(display_img, (u, v), 2, (0, 0, 255), 3)
                        cv2.putText(display_img, f"Ball {i+1}", (u-20, v-20), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                self.root.after(0, self.update_cam_image, display_img)
        finally:
            self.pipeline.stop()

    def get_target_coordinates(self):
        """인식 버튼을 눌렀을 때 실행되는 정밀 원 검출 로직"""
        frames = self.pipeline.wait_for_frames()
        aligned = self.align.process(frames)
        color_f = aligned.get_color_frame()
        depth_f = aligned.get_depth_frame()
        if not color_f or not depth_f: return None
        
        img = np.asanyarray(color_f.get_data())
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(cv2.GaussianBlur(img, (11, 11), 0), cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([85, 100, 100]), np.array([105, 255, 255]))
        masked_gray = cv2.bitwise_and(gray, gray, mask=mask)

        circles = cv2.HoughCircles(
            masked_gray, 
            cv2.HOUGH_GRADIENT, dp=1.2, minDist=40, 
            param1=50, param2=25, minRadius=20, maxRadius=60
        )

        if circles is not None:
            circles = np.uint16(np.around(circles))
            # 여러 개가 보일 경우 첫 번째 공(0번)을 타겟으로 설정
            u, v, r = circles[0][0]
            
            depth = depth_f.get_distance(int(u), int(v)) * 1000
            if depth > 0:
                profile = color_f.get_profile().as_video_stream_profile()
                intrinsics = profile.get_intrinsics()
                p_cam = rs.rs2_deproject_pixel_to_point(intrinsics, [u, v], depth)
                dst_pt = cv2.transform(np.array([[[p_cam[0], p_cam[1], p_cam[2]]]], dtype=np.float32), self.matrix)
                return [dst_pt[0][0][0], dst_pt[0][0][1] + Y_OFFSET, dst_pt[0][0][2], 0, 180, 0]
        return None

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

    def scan_target(self):
        target = self.get_target_coordinates()
        if target:
            self.current_target = target
            self.add_log(f"🎯 타겟 고정: X={target[0]:.1f}, Y={target[1]:.1f}")
        else:
            self.add_log("❌ 공을 찾을 수 없습니다.")

    def go_target_top(self):
        if not self.current_target: return
        tx, ty, _, rx, ry, rz = self.current_target
        self.send_command(f"MOVE,{tx:.2f},{ty:.2f},{SAFE_APPROACH[2]},{rx},{ry},{rz}")

    def down_and_grip(self):
        if not self.current_target: return
        tx, ty, tz, rx, ry, rz = self.current_target
        self.send_command(f"DOWN,{tx:.2f},{ty:.2f},{tz:.2f},{rx},{ry},{rz}")
        time.sleep(0.3)
        self.send_command("GRIP")
        time.sleep(0.5) # 그립 후 대기
        self.send_command(f"MOVE,{tx:.2f},{ty:.2f},{SAFE_APPROACH[2]},{rx},{ry},{rz}")

    def delivery_path(self):
        self.send_command(f"MOVE,{HOME_POS[0]},{HOME_POS[1]},{HOME_POS[2]},{HOME_POS[3]},{HOME_POS[4]},{HOME_POS[5]}")
        self.send_command("RELEASE")

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

if __name__ == "__main__":
    root = tk.Tk()
    app = BallPickupDetailControl(root)
    root.mainloop()