import socket
import cv2
import cv2.aruco as aruco
import numpy as np
import threading
import time
import customtkinter as ctk
from PIL import Image

# [1. 2D 캘리브레이션 데이터 설정]
ROBOT_PTS = np.array([
    [705.36, 836.62],
    [281.38, 836.63],
    [1186.74, 836.62],
    [682.55, 386.54],
    [682.54, 1186.51]
], dtype=np.float32)

CAM_PTS = np.array([
    [-29.69, 73.95],
    [-107.16, -452.38],
    [89.68, 340.21],
    [-501.36, 26.53],
    [422.36, -155.51]
], dtype=np.float32)

# 아핀 변환 행렬 계산
M_affine, _ = cv2.estimateAffine2D(CAM_PTS, ROBOT_PTS)

# [설정 값 고정]
MOVE_Z_DEPTH = 1100.0 
FIXED_ORIENTATION = "77.79,88.72,91.05" # 요청하신 Rx, Ry, Rz 값

# 경로 설정
WAYPOINT_SAFE = "MOVE,317.08,1116.89,1450.00,166.5,-43.72,-139.19" 
FINAL_SAFE = "MOVE,346.08,949.89,1268.99,20.55,71.54,105.80"

class RobotApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Doosan Robot Master - Fixed Orientation Tracking")
        self.geometry("1200x850")
        
        self.conn = None
        self.last_target_pos = None 
        
        self.camera_matrix = np.array([[900.0, 0, 640.0], [0, 900.0, 360.0], [0, 0, 1]], dtype=float)
        self.dist_coeffs = np.zeros((5, 1))
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        self.params = aruco.DetectorParameters()
        
        self.setup_ui()
        self.setup_camera()
        self.start_socket_server()

    def setup_ui(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.sidebar = ctk.CTkFrame(self, width=220, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        
        ctk.CTkLabel(self.sidebar, text="ROBOT MASTER", font=("Arial", 20, "bold")).pack(pady=30)

        self.btn_home = ctk.CTkButton(self.sidebar, text="🏠 홈 이동", command=lambda: self.send_command("MOVE_HOME,"))
        self.btn_home.pack(pady=10, padx=20)

        self.btn_safe = ctk.CTkButton(self.sidebar, text="🛡️ 안전위치 이동", fg_color="#34495e", command=self.start_safe_sequence)
        self.btn_safe.pack(pady=10, padx=20)

        self.btn_stir = ctk.CTkButton(self.sidebar, text="🌀 마커 추적 및 이동", 
                                      fg_color="#2ECC71", command=self.execute_marker_move)
        self.btn_stir.pack(pady=10, padx=20)

        self.status_label = ctk.CTkLabel(self.sidebar, text="연결 대기 중...", text_color="#E67E22")
        self.status_label.pack(side="bottom", pady=30)

        self.video_container = ctk.CTkFrame(self, fg_color="#1a1a1a")
        self.video_container.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        self.video_label = ctk.CTkLabel(self.video_container, text="")
        self.video_label.pack(expand=True)

    def setup_camera(self):
        self.cap = cv2.VideoCapture(1)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.update_video()

    def update_video(self):
        ret, frame = self.cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = aruco.detectMarkers(gray, self.aruco_dict, parameters=self.params)
            
            if ids is not None:
                aruco.drawDetectedMarkers(frame, corners, ids)
                rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, 100, self.camera_matrix, self.dist_coeffs)
                
                cam_x, cam_y = tvecs[0][0][0], tvecs[0][0][1]
                
                input_pt = np.array([cam_x, cam_y, 1.0])
                robot_xy = np.dot(M_affine, input_pt)
                self.last_target_pos = robot_xy
                
                cv2.putText(frame, f"Target X:{robot_xy[0]:.1f} Y:{robot_xy[1]:.1f}", 
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
        if not self.conn:
            self.status_label.configure(text="❌ 연결 없음", text_color="#E74C3C")
            return
        if self.last_target_pos is None:
            self.status_label.configure(text="🔍 마커 미감지", text_color="#E67E22")
            return
        threading.Thread(target=self._marker_move_worker, daemon=True).start()

    def _marker_move_worker(self):
        rx, ry = self.last_target_pos
        self.status_label.configure(text=f"🚀 이동: {rx:.1f}, {ry:.1f}", text_color="#3498db")
        
        # [수정된 부분] 회전값 FIXED_ORIENTATION 적용
        move_cmd = f"MOVE,{rx:.2f},{ry:.2f},{MOVE_Z_DEPTH:.2f},{FIXED_ORIENTATION}"
        
        if self.send_command_and_wait(move_cmd):
            self.status_label.configure(text="✅ 도착 및 정지", text_color="#1ABC9C")
        else:
            self.status_label.configure(text="❌ 이동 실패", text_color="#E74C3C")

    def start_safe_sequence(self):
        if not self.conn: return
        threading.Thread(target=self._safe_move_worker, daemon=True).start()

    def _safe_move_worker(self):
        self.status_label.configure(text="🛡️ 안전위치 시퀀스...", text_color="#3498db")
        if self.send_command_and_wait(WAYPOINT_SAFE):
            time.sleep(0.1)
            if self.send_command_and_wait(FINAL_SAFE):
                self.status_label.configure(text="✅ 안전위치 도착", text_color="#1ABC9C")

    def send_command_and_wait(self, cmd):
        if not self.conn: return False
        try:
            self.conn.setblocking(False)
            try:
                while self.conn.recv(1024): pass
            except: pass
            self.conn.setblocking(True)
            self.conn.sendall((cmd + "\n").encode())
            self.conn.settimeout(15.0) 
            raw_data = self.conn.recv(1024).decode().strip()
            return "DONE" in raw_data
        except:
            return False
        finally:
            self.conn.settimeout(None)

    def start_socket_server(self):
        def server_thread():
            server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind(("0.0.0.0", 30002))
            server.listen(1)
            while True:
                conn, addr = server.accept()
                self.conn = conn
                self.status_label.configure(text=f"✅ 로봇 연결됨", text_color="#1ABC9C")
        threading.Thread(target=server_thread, daemon=True).start()

    def send_command(self, cmd):
        if self.conn:
            try:
                self.conn.sendall((cmd + "\n").encode())
                return True
            except:
                self.conn = None
                return False
        return False

if __name__ == "__main__":
    app = RobotApp()
    app.mainloop()