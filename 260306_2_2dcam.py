import socket
import cv2
import cv2.aruco as aruco
import numpy as np
import threading
import time
import customtkinter as ctk
from PIL import Image

# [1. 최신 3점 데이터 기반 매핑]
# 로봇 좌표 (X, Y)
ROBOT_PTS = np.array([
    [727.38, 849.51],  # 중심
    [927.41, 849.51],  # 동쪽 (+X)
    [727.46, 1049.52]  # 북쪽 (+Y)
], dtype=np.float32)

# 카메라 좌표 (X, Y)
CAM_PTS = np.array([
    [-61.0, 105.0],
    [35.57, 287.56],
    [208.84, 21.30]
], dtype=np.float32)

# 3개의 점만 있으므로 estimateAffine2D 대신 getAffineTransform 사용 (정확히 일치시킴)
M_affine = cv2.getAffineTransform(CAM_PTS, ROBOT_PTS)

# [2. 고정 설정]
MOVE_Z_DEPTH = 1100.0 
FIXED_ORIENTATION = "77.79,88.72,91.05"

class RobotApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Doosan Robot Master - Precise 3-Point Cal")
        self.geometry("1200x850")
        
        self.conn = None
        self.last_target_pos = None 
        
        # 카메라 파라미터 (동일)
        self.camera_matrix = np.array([[900.0, 0, 640.0], [0, 900.0, 360.0], [0, 0, 1]], dtype=float)
        self.dist_coeffs = np.zeros((5, 1))
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        self.params = aruco.DetectorParameters()
        
        self.setup_ui()
        self.setup_camera()
        self.start_socket_server()

    def update_video(self):
        ret, frame = self.cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = aruco.detectMarkers(gray, self.aruco_dict, parameters=self.params)
            
            if ids is not None:
                aruco.drawDetectedMarkers(frame, corners, ids)
                _, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, 100, self.camera_matrix, self.dist_coeffs)
                
                # 카메라 좌표 추출
                cx, cy = tvecs[0][0][0], tvecs[0][0][1]
                
                # [변환] 카메라 (cx, cy) -> 로봇 (rx, ry)
                # M_affine: [[a, b, tx], [c, d, ty]] 형태
                rx = M_affine[0,0]*cx + M_affine[0,1]*cy + M_affine[0,2]
                ry = M_affine[1,0]*cx + M_affine[1,1]*cy + M_affine[1,2]
                
                self.last_target_pos = [rx, ry]
                
                # 화면에 계산된 로봇 좌표 표시
                cv2.putText(frame, f"Robot Target X:{rx:.1f} Y:{ry:.1f}", 
                            (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            else:
                self.last_target_pos = None

            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img)
            cw, ch = self.video_container.winfo_width(), self.video_container.winfo_height()
            if cw > 10 and ch > 10:
                ratio = 1280/720
                nw, nh = (cw, int(cw/ratio)) if cw/ch < ratio else (int(ch*ratio), ch)
                img_tk = ctk.CTkImage(light_image=img_pil, dark_image=img_pil, size=(nw, nh))
                self.video_label.configure(image=img_tk)
        
        self.after(10, self.update_video)

    def execute_marker_move(self):
        if not self.conn or self.last_target_pos is None: return
        threading.Thread(target=self._marker_move_worker, daemon=True).start()

    def _marker_move_worker(self):
        rx, ry = self.last_target_pos
        move_cmd = f"MOVE,{rx:.2f},{ry:.2f},{MOVE_Z_DEPTH:.2f},{FIXED_ORIENTATION}"
        self.status_label.configure(text=f"🚀 이동 중: {rx:.1f}, {ry:.1f}")
        
        if self.send_command_and_wait(move_cmd):
            self.status_label.configure(text="✅ 도착 성공", text_color="#1ABC9C")

    # (이하 생략: send_command_and_wait, setup_ui, setup_camera 등은 이전과 동일)
    def setup_ui(self):
        self.grid_columnconfigure(1, weight=1); self.grid_rowconfigure(0, weight=1)
        self.sidebar = ctk.CTkFrame(self, width=220, corner_radius=0); self.sidebar.grid(row=0, column=0, sticky="nsew")
        ctk.CTkLabel(self.sidebar, text="ROBOT MASTER", font=("Arial", 20, "bold")).pack(pady=30)
        self.btn_stir = ctk.CTkButton(self.sidebar, text="🌀 마커 추적 및 이동", fg_color="#2ECC71", command=self.execute_marker_move); self.btn_stir.pack(pady=10, padx=20)
        self.status_label = ctk.CTkLabel(self.sidebar, text="연결 대기 중...", text_color="#E67E22"); self.status_label.pack(side="bottom", pady=30)
        self.video_container = ctk.CTkFrame(self, fg_color="#1a1a1a"); self.video_container.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        self.video_label = ctk.CTkLabel(self.video_container, text=""); self.video_label.pack(expand=True)

    def setup_camera(self):
        self.cap = cv2.VideoCapture(1); self.cap.set(3, 1280); self.cap.set(4, 720); self.update_video()

    def start_socket_server(self):
        def server_thread():
            server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind(("0.0.0.0", 30002)); server.listen(1)
            while True:
                conn, addr = server.accept(); self.conn = conn
                self.status_label.configure(text=f"✅ 로봇 연결됨", text_color="#1ABC9C")
        threading.Thread(target=server_thread, daemon=True).start()

    def send_command_and_wait(self, cmd):
        if not self.conn: return False
        try:
            self.conn.sendall((cmd + "\n").encode())
            self.conn.settimeout(15.0); raw = self.conn.recv(1024).decode()
            return "DONE" in raw
        except: return False

if __name__ == "__main__":
    app = RobotApp(); app.mainloop()