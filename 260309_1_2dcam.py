import socket
import cv2
import numpy as np
import threading
import customtkinter as ctk
from PIL import Image
import time

# [1. 호모그래피 데이터]
CAM_PTS = np.array([[190, -324], [-440, 80], [270, -50], [-27, 322]], dtype=np.float32)
ROBOT_PTS = np.array([[374, 782], [744, 424], [744, 1137], [1110, 789]], dtype=np.float32)
H_matrix, _ = cv2.findHomography(CAM_PTS, ROBOT_PTS)

# [2. 오프셋 설정 (제시해주신 P1-P2-P3 차이값)]
# P1(기준): 720.07, 560.14, 1018.20, 78.37, 85.31, 92.49
# P2(1단계): 904.74, 556.07, 790.51, 83.47, 91.29, 132.80
D1 = {'x': 184.67, 'y': -4.07, 'z': -227.69, 'rx': 83.47, 'ry': 91.29, 'rz': 132.80}

# P3(2단계): 873.05, 564.69, 886.61, 86.29, 87.58, 103.19
D2 = {'x': 152.98, 'y': 4.55, 'z': -131.59, 'rx': 86.29, 'ry': 87.58, 'rz': 103.19}

FIXED_ORIENTATION = "78.37,85.31,92.49" 
MOVE_Z_DEPTH = 1018.20 

CROP_X, CROP_Y, CROP_W, CROP_H = 300, 100, 700, 550
COLOR_RANGES = {
    "BLUE":   {"low": [90, 100, 100],  "high": [125, 255, 255], "color": (255, 0, 0)},
    "ORANGE": {"low": [10, 150, 150],  "high": [25, 255, 255],  "color": (0, 165, 255)},
    "YELLOW": {"low": [25, 100, 100],  "high": [35, 255, 255],  "color": (0, 255, 255)},
    "PURPLE": {"low": [130, 50, 50],   "high": [160, 255, 255], "color": (255, 0, 255)}
}

class RobotApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Doosan Robot - Multi-Step Sequence Control")
        self.geometry("1300x850")
        self.conn = None
        self.cap = None
        self.last_target_pos = None 
        self.target_color_key = "BLUE"
        
        self.setup_ui()
        self.setup_camera()
        self.start_socket_server()

    def setup_camera(self):
        """카메라 연결 문제 해결을 위한 초기화 로직"""
        if self.cap is not None:
            self.cap.release()
            time.sleep(0.5) # 장치 해제 대기

        # 1번 인덱스 우선 시도, 실패 시 0번 시도
        for idx in [1, 0, 2]:
            self.cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            if self.cap.isOpened():
                print(f"카메라 {idx}번 연결 성공")
                break
        
        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) # 프레임 지연 방지
            self.update_video()
            self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1) 
        
            # 수동 초점 값 설정 (0 ~ 255 범위, 웹캠 모델마다 다름)
            # 보통 0은 무한대, 숫자가 커질수록 가까운 곳에 초점이 맞습니다.
            # 30~60 사이의 값을 먼저 시도해 보세요.
            self.cap.set(cv2.CAP_PROP_FOCUS, 200)
        else:
            self.status_label.configure(text="⚠️ Camera Error", text_color="#E74C3C")

    def update_video(self):
        if self.cap is None or not self.cap.isOpened():
            return

        ret, frame = self.cap.read()
        if ret:
            cropped = frame[CROP_Y:CROP_Y+CROP_H, CROP_X:CROP_X+CROP_W]
            hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
            color_cfg = COLOR_RANGES[self.target_color_key]
            mask = cv2.inRange(hsv, np.array(color_cfg["low"]), np.array(color_cfg["high"]))
            mask = cv2.dilate(cv2.erode(mask, None, iterations=2), None, iterations=2)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if len(contours) > 0:
                c = max(contours, key=cv2.contourArea)
                ((x, y), radius) = cv2.minEnclosingCircle(c)
                if radius > 15:
                    full_cx, full_cy = int(x) + CROP_X, int(y) + CROP_Y
                    cx_raw = (full_cx - 640) * (500 / 900)
                    cy_raw = (full_cy - 360) * (500 / 900)
                    
                    input_pt = np.array([[[cx_raw, cy_raw]]], dtype=np.float32)
                    robot_pt = cv2.perspectiveTransform(input_pt, H_matrix)
                    self.last_target_pos = [robot_pt[0][0][0], robot_pt[0][0][1]]
                    
                    cv2.circle(frame, (full_cx, full_cy), int(radius), color_cfg["color"], 2)
                    cv2.putText(frame, f"X:{self.last_target_pos[0]:.1f} Y:{self.last_target_pos[1]:.1f}", 
                                (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img)
            img_tk = ctk.CTkImage(light_image=img_pil, dark_image=img_pil, size=(900, 500))
            self.video_label.configure(image=img_tk)
        
        self.after(20, self.update_video)

    def setup_ui(self):
        self.grid_columnconfigure(1, weight=1); self.grid_rowconfigure(0, weight=1)
        self.sidebar = ctk.CTkFrame(self, width=250, corner_radius=0); self.sidebar.grid(row=0, column=0, sticky="nsew")
        ctk.CTkLabel(self.sidebar, text="STEP SEQUENCE", font=("Arial", 20, "bold")).pack(pady=20)
        
        ctk.CTkButton(self.sidebar, text="🔄 RECONNECT CAM", command=self.setup_camera, fg_color="#34495e").pack(pady=5, padx=20)
        
        for color in COLOR_RANGES.keys():
            btn_color = "#3498db" if color=="BLUE" else "#e67e22" if color=="ORANGE" else "#f1c40f" if color=="YELLOW" else "#9b59b6"
            ctk.CTkButton(self.sidebar, text=f"{color} TRACK", fg_color=btn_color, command=lambda c=color: self.set_target_color(c)).pack(pady=5, padx=20)

        ctk.CTkButton(self.sidebar, text="🚀 TRACK MOVE", fg_color="#2ecc71", font=("Arial", 14, "bold"), height=50, command=self.execute_marker_move).pack(pady=20, padx=20)
        ctk.CTkButton(self.sidebar, text="🥄 DUAL SCOOP", fg_color="#e74c3c", font=("Arial", 14, "bold"), height=50, command=self.execute_dual_scoop).pack(pady=10, padx=20)
        
        self.status_label = ctk.CTkLabel(self.sidebar, text="Offline", text_color="#E67E22"); self.status_label.pack(side="bottom", pady=20)
        self.video_container = ctk.CTkFrame(self, fg_color="#1a1a1a"); self.video_container.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        self.video_label = ctk.CTkLabel(self.video_container, text=""); self.video_label.pack(expand=True)

    def start_socket_server(self):
        def server_thread():
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("0.0.0.0", 30002)); s.listen(1)
            while True:
                conn, addr = s.accept(); self.conn = conn
                self.status_label.configure(text=f"✅ Robot Connected", text_color="#1ABC9C")
        threading.Thread(target=server_thread, daemon=True).start()

    def send_command_and_wait(self, cmd):
        if not self.conn: return False
        try:
            self.conn.sendall((cmd + "\n").encode())
            self.conn.settimeout(15.0)
            return "DONE" in self.conn.recv(1024).decode()
        except: return False

    def _move_worker(self, mode):
        bx, by = self.last_target_pos
        
        if mode == "TRACK":
            cmd = f"MOVE,{bx:.2f},{by:.2f},{MOVE_Z_DEPTH:.2f},{FIXED_ORIENTATION}"
            self.send_command_and_wait(cmd)
        
        elif mode == "DUAL_STEP":
            # --- 1단계: P2 지점으로 이동 ---
            tx1, ty1, tz1 = bx + D1['x'], by + D1['y'], MOVE_Z_DEPTH + D1['z']
            self.status_label.configure(text="🔄 Moving to P2...", text_color="#F1C40F")
            cmd1 = f"MOVE,{tx1:.2f},{ty1:.2f},{tz1:.2f},{D1['rx']:.2f},{D1['ry']:.2f},{D1['rz']:.2f}"
            
            if self.send_command_and_wait(cmd1):
                # --- 2단계: P3 지점으로 이동 ---
                tx2, ty2, tz2 = bx + D2['x'], by + D2['y'], MOVE_Z_DEPTH + D2['z']
                self.status_label.configure(text="🔄 Moving to P3...", text_color="#F1C40F")
                cmd2 = f"MOVE,{tx2:.2f},{ty2:.2f},{tz2:.2f},{D2['rx']:.2f},{D2['ry']:.2f},{D2['rz']:.2f}"
                self.send_command_and_wait(cmd2)
                
            self.status_label.configure(text="✅ Complete", text_color="#1ABC9C")

    def execute_marker_move(self):
        if self.conn and self.last_target_pos:
            threading.Thread(target=self._move_worker, args=("TRACK",), daemon=True).start()

    def execute_dual_scoop(self):
        if self.conn and self.last_target_pos:
            threading.Thread(target=self._move_worker, args=("DUAL_STEP",), daemon=True).start()

    def set_target_color(self, color_key): self.target_color_key = color_key

if __name__ == "__main__":
    app = RobotApp(); app.mainloop()