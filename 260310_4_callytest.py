import customtkinter as ctk
import cv2
import numpy as np
import socket
import threading
import time
from PIL import Image, ImageTk

# 1. 저장된 데이터 로드
try:
    calib_data = np.load('calibration_data.npz')
    # mtx = calib_data['mtx'] # 렌즈 보정은 쓰지 않으므로 주석 처리
    
    H_matrix = np.load('homography_matrix.npy')
    print("✅ 캘리브레이션 데이터 로드 성공!")
except:
    print("❌ 필요한 데이터 파일(.npz 또는 .npy)이 없습니다.")
    exit()

# 2. 로봇 설정
Z_HEIGHT = 1018.20 
BASE_ORI = "78.37,85.31,92.49"

# 3. HSV 색상 범위 설정 (실제 환경에 맞춰 미세 조정 필요)
# [Lower_HSV, Upper_HSV, Display_Color(BGR)]
COLOR_RANGES = {
    "SKY":   ([90, 100, 100],  [110, 255, 255], (255, 255, 0)),  # 하늘색
    "YELLOW":([20, 100, 100],  [30, 255, 255],  (0, 255, 255)),  # 노란색
    "ORANGE":([10, 100, 100],  [20, 255, 255],  (0, 165, 255)),  # 주황색
    "PURPLE":([130, 50, 50],   [160, 255, 255], (128, 0, 128))   # 진한 보라색
}

class ColorTrackerRobotApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # GUI 설정
        self.title("Doosan Robot Color Tracker")
        self.geometry("1400x800")
        
        # 카메라 & 통신 설정
        self.cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.conn = None
        self.is_connected = False
        self.current_track_color = None # 현재 추적 중인 색상
        self.last_send_time = time.time()
        
        self.start_socket_server() # 로봇 접속 대기

        # 레이아웃 구성
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=3)
        self.grid_rowconfigure(0, weight=1)

        # 왼쪽 제어 패널
        self.control_panel = ctk.CTkFrame(self)
        self.control_panel.grid(row=0, column=0, padx=20, pady=20, sticky="ns")
        
        self.status_label = ctk.CTkLabel(self.control_panel, text="Wait Robot...", text_color="red", font=("Arial", 20, "bold"))
        self.status_label.pack(pady=30)
        
        # 색상 추적 버튼들
        colors = [
            ("Sky Blue", "SKY", "#87CEEB"),
            ("Yellow", "YELLOW", "#FFD700"),
            ("Orange", "ORANGE", "#FF8C00"),
            ("Dark Purple", "PURPLE", "#800080")
        ]
        
        for text, key, color_hex in colors:
            btn = ctk.CTkButton(self.control_panel, text=f"Track {text}",
                                fg_color=color_hex, text_color="black" if key != "PURPLE" else "white",
                                font=("Arial", 16, "bold"), height=50,
                                command=lambda k=key: self.set_track_color(k))
            btn.pack(pady=15, fill="x", padx=20)
            
        # 추적 중지 버튼
        self.stop_btn = ctk.CTkButton(self.control_panel, text="Stop Tracking", fg_color="gray",
                                      font=("Arial", 16, "bold"), height=50, command=self.stop_tracking)
        self.stop_btn.pack(pady=30, fill="x", padx=20)

        # 오른쪽 카메라 화면 패널
        self.video_label = ctk.CTkLabel(self, text="Camera Loading...")
        self.video_label.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        
        self.update_video() # 비디오 업데이트 루프 시작

    def start_socket_server(self):
        def server_thread():
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("0.0.0.0", 30002))
            s.listen(1)
            print("🚀 로봇 접속 대기 중...")
            conn, addr = s.accept()
            self.conn = conn
            self.is_connected = True
            self.status_label.configure(text=f"Robot Connected: {addr[0]}", text_color="green")
            print(f"✅ 로봇 연결됨: {addr}")
        threading.Thread(target=server_thread, daemon=True).start()

    def set_track_color(self, color_key):
        self.current_track_color = color_key
        print(f"🎯 추적 시작: {color_key}")

    def stop_tracking(self):
        self.current_track_color = None
        print("⏹️ 추적 중지")

    def convert_and_send(self, cx, cy):
        # 0.5초 간격 전송 제한
        if time.time() - self.last_send_time < 0.5:
            return
            
        if self.conn and self.is_connected:
            # 카메라 좌표 전처리 (캘리브레이션 시와 동일한 수식)
            raw_x = (cx - 640) * (500 / 900)
            raw_y = (cy - 360) * (500 / 900)
            
            # 호모그래피 변환
            input_pt = np.array([[[raw_x, raw_y]]], dtype=np.float32)
            robot_pt = cv2.perspectiveTransform(input_pt, H_matrix)
            rx, ry = robot_pt[0][0][0], robot_pt[0][0][1]
            
            # 로봇으로 전송
            cmd = f"MOVE,{rx:.2f},{ry:.2f},{Z_HEIGHT:.2f},{BASE_ORI}\n"
            try:
                self.conn.sendall(cmd.encode())
                self.last_send_time = time.time()
                print(f"📡 {self.current_track_color} 공으로 이동: X:{rx:.1f}, Y:{ry:.1f}")
            except Exception as e:
                print(f"❌ 전송 에러: {e}")

    def update_video(self):
        ret, frame = self.cap.read()
        if ret:
            # 캘리브레이션 때 'return frame'을 쓰셨으므로 원본(frame) 사용
            display_frame = frame.copy()
            
            # 4. 색상 추적 로직
            if self.current_track_color:
                lower, upper, draw_color = COLOR_RANGES[self.current_track_color]
                
                # HSV 변환 및 마스크 생성
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
                
                # 노이즈 제거 (Opening/Closing)
                kernel = np.ones((5,5), np.uint8)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                
                # 윤곽선 찾기
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    # 가장 큰 덩어리 찾기
                    c = max(contours, key=cv2.contourArea)
                    if cv2.contourArea(c) > 500: # 최소 크기 제한
                        # 중심점 계산
                        M = cv2.moments(c)
                        if M["m00"] != 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])
                            
                            # 화면에 표시
                            cv2.drawContours(display_frame, [c], -1, draw_color, 3)
                            cv2.circle(display_frame, (cx, cy), 7, (0, 0, 255), -1)
                            cv2.putText(display_frame, f"Tracking: {self.current_track_color}", (cx+20, cy-20),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, draw_color, 2)
                            
                            # 로봇 좌표 변환 및 전송
                            self.convert_and_send(cx, cy)
            
            # OpenCV 이미지를 Tkinter용 이미지로 변환
            cv2image = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGBA)
            img = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
            
        # 10ms마다 비디오 업데이트
        self.after(10, self.update_video)

if __name__ == "__main__":
    # CustomTkinter 테마 설정
    ctk.set_appearance_mode("Dark") # "System" (standard), "Dark", "Light"
    ctk.set_default_color_theme("blue") # Themes: "blue" (standard), "green", "dark-blue"
    
    app = ColorTrackerRobotApp()
    app.mainloop()