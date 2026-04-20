import socket
import cv2
import numpy as np
import threading
import customtkinter as ctk
from PIL import Image, ImageTk
import time
import os

# [1. 설정 데이터 및 경로]
H_MATRIX_PATH = "homography_matrix.npy"
HOME_POSE = "535.65,-246.04,786.67,3.83,124.54,175.95"
DISCHARGE_POSE = "823.70,708.96,962.39,39.90,90.00,170.00"
SAFE_POSE = "770.47,793.10,683.21,73.44,124.54,175.84"
FIXED_ORIENTATION = "127.01,165.08,-14.92" 
MOVE_Z_DEPTH = 807.20 

X_MIN, X_MAX = 472, 1052
Y_MIN, Y_MAX = 376, 1145

# 구역별 오프셋 데이터 (기존 유지)
AREA_A = [
    {'x': 3.80,  'y': -6.40,  'z': -71.10,  'rx': 77.67, 'ry': 102.78, 'rz': 91.20},
    {'x': 4.70,  'y': 51.80,  'z': -203.20, 'rx': 80.18, 'ry': 112.85, 'rz': 88.94},
    {'x': 62.60, 'y': 97.90,  'z': -183.20, 'rx': 95.37, 'ry': 100.90, 'rz': 96.70},
    {'x': 31.50, 'y': -35.50, 'z': 38.50,   'rx': 89.75, 'ry': 85.50, 'rz': 94.60}
]
AREA_B = [
    {'x': 55.4, 'y': -23.1, 'z': -332.5, 'rx': 98.8, 'ry': 65.4, 'rz': 91.7},
    {'x': 99.8, 'y': -282.4, 'z': -333.8, 'rx': 99.5, 'ry': 64.7, 'rz': 94.8}, 
    {'x': 99.6, 'y': -119.6, 'z': -176.3, 'rx': 97.6, 'ry': 88.5, 'rz': 97.5}, 
    {'x': 40.0,  'y': -105.0, 'z': 110.0,  'rx': 103.0, 'ry': 80.0, 'rz': 92.0}
]
AREA_C = [
    {'x': 29.80,  'y': -0.20,  'z': -123.60, 'rx': 79.50, 'ry': 86.90, 'rz': 129.40},
    {'x': 83.10,  'y': 0.70,   'z': -257.70, 'rx': 79.60, 'ry': 91.10, 'rz': 129.30},
    {'x': 100.80, 'y': -20.90, 'z': -203.30, 'rx': 78.70, 'ry': 88.40, 'rz': 113.50},
    {'x': 66.70,  'y': -38.10, 'z': -61.50,  'rx': 80.10, 'ry': 83.70, 'rz': 101.50}
]

CROP_X, CROP_Y, CROP_W, CROP_H = 250, 150, 550, 450

COLOR_RANGES = {
    "BLUE":   {"low": [90, 100, 100],   "high": [125, 255, 255], "color": (255, 0, 0)},
    "ORANGE": {"low": [5, 120, 100],  "high": [20, 255, 255],  "color": (0, 165, 255)},
    "YELLOW": {"low": [22, 70, 100],  "high": [35, 255, 255],  "color": (0, 255, 255)},
    "PURPLE": {"low": [130, 50, 50],  "high": [165, 255, 255], "color": (255, 0, 255)}
}

class RobotApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Doosan Robot Kitchen - Stable Control System")
        self.geometry("1400x900")
        
        self.load_homography()
        self.conn = None
        self.cap = None
        self.last_target_pos = None
        self.current_assigned_pos = None
        self.target_color_key = "BLUE"
        self.is_moving = False
        self.is_all_auto = False
        self.stop_requested = False
        self.setup_ui()
        self.setup_camera()
        self.start_socket_server()

    def load_homography(self):
        global H_matrix
        if os.path.exists(H_MATRIX_PATH):
            H_matrix = np.load(H_MATRIX_PATH)
        else:
            H_matrix = None

    def setup_ui(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.sidebar = ctk.CTkFrame(self, width=300, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        
        ctk.CTkLabel(self.sidebar, text="🤖 KITCHEN MASTER", font=("Arial", 22, "bold")).pack(pady=20)

        btn_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        btn_frame.pack(pady=10, padx=20, fill="x")
        ctk.CTkButton(btn_frame, text="🏠 HOME", fg_color="#2980b9", font=("Arial", 14, "bold"), 
                      height=45, command=self.execute_home_move).pack(side="left", expand=True, padx=5)
        ctk.CTkButton(btn_frame, text="🛡 SAFE", fg_color="#27ae60", font=("Arial", 12, "bold"), 
                      height=40, command=self.execute_safe_move).pack(side="left", expand=True, padx=2)
        ctk.CTkButton(btn_frame, text="📤 DISCHARGE", fg_color="#8e44ad", font=("Arial", 14, "bold"), 
                      height=45, command=self.execute_discharge_move).pack(side="left", expand=True, padx=5)

        ctk.CTkLabel(self.sidebar, text="--- SEQUENCES ---", font=("Arial", 12)).pack(pady=5)
        self.discharge_seq_btn = ctk.CTkButton(
            self.sidebar, text="🔥 DISCHARGE SEQ", fg_color="#27ae60", font=("Arial", 16, "bold"), 
            height=50, command=self.execute_discharge_sequence
        )
        self.discharge_seq_btn.pack(pady=5, padx=20, fill="x")

        # [추가] SCREW 시퀀스 버튼
        self.screw_btn = ctk.CTkButton(
            self.sidebar, text="🌀 SCREW (STIR)", fg_color="#16a085", font=("Arial", 16, "bold"), 
            height=50, command=self.execute_screw_move
        )
        self.call_sub_btn = ctk.CTkButton(self.sidebar, text="⚡ ddischarge", fg_color="#e67e22", 
                                         font=("Arial", 14, "bold"), height=45, command=self.execute_sub_program_call)
        self.call_sub_btn.pack(pady=3, padx=20, fill="x")
        ctk.CTkButton(self.sidebar, text="📥 BADE INPUT", fg_color="#34495e", 
                      font=("Arial", 14, "bold"), height=45, 
                      command=lambda: self.execute_named_sub_program("badeinput")).pack(pady=3, padx=20, fill="x")

        ctk.CTkButton(self.sidebar, text="🔥 FRY BADE", fg_color="#c0392b", 
                      font=("Arial", 14, "bold"), height=45, 
                      command=lambda: self.execute_named_sub_program("frybade")).pack(pady=3, padx=20, fill="x")

        ctk.CTkButton(self.sidebar, text="📤 FRY OUTPUT", fg_color="#2980b9", 
                      font=("Arial", 14, "bold"), height=45, 
                      command=lambda: self.execute_named_sub_program("fryoutput")).pack(pady=3, padx=20, fill="x")
        ctk.CTkLabel(self.sidebar, text="--- COLOR MODE ---", font=("Arial", 12)).pack(pady=5)
        for color in COLOR_RANGES.keys():
            btn_color = "#3498db" if color=="BLUE" else "#e67e22" if color=="ORANGE" else "#f1c40f" if color=="YELLOW" else "#9b59b6"
            ctk.CTkButton(self.sidebar, text=f"{color} MODE", fg_color=btn_color, 
                          text_color="white" if color != "YELLOW" else "black",
                          command=lambda c=color: self.set_target_color(c)).pack(pady=3, padx=20, fill="x")

        ctk.CTkLabel(self.sidebar, text="--- ACTIONS ---", font=("Arial", 12)).pack(pady=10)
        ctk.CTkButton(self.sidebar, text="🚀 TRACK MOVE", fg_color="#27ae60", font=("Arial", 16, "bold"), 
                      height=50, command=self.execute_marker_move).pack(pady=5, padx=20, fill="x")
        ctk.CTkButton(self.sidebar, text="🔨 TRIPLE STRIKE", fg_color="#d35400", font=("Arial", 16, "bold"), 
                      height=60, command=self.execute_strike_move).pack(pady=10, padx=20, fill="x")
        ctk.CTkButton(self.sidebar, text="🥄 DUAL SCOOP", fg_color="#c0392b", font=("Arial", 16, "bold"), 
                      height=50, command=self.execute_dual_scoop).pack(pady=5, padx=20, fill="x")

        self.status_label = ctk.CTkLabel(self.sidebar, text="Offline", text_color="#E67E22", font=("Arial", 14))
        self.status_label.pack(side="bottom", pady=20)

        self.video_container = ctk.CTkFrame(self, fg_color="#121212")
        self.video_container.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        self.video_label = ctk.CTkLabel(self.video_container, text="")
        self.video_label.pack(expand=True)
        self.all_auto_btn = ctk.CTkButton(
        self.sidebar, text="🤖 ALL-AUTO START", fg_color="#e74c3c", font=("Arial", 18, "bold"), 
        height=60, command=self.toggle_all_auto
        )
        self.all_auto_btn.pack(pady=15, padx=20, fill="x")
    # 카메라 및 서버 설정 (기존 유지)
    def setup_camera(self):
        if self.cap is not None: self.cap.release()
        for idx in [1, 0, 2]:
            self.cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            if self.cap.isOpened(): break
        if self.cap.isOpened():
            self.cap.set(3, 1280); self.cap.set(4, 720)
            self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1); #self.cap.set(cv2.CAP_PROP_FOCUS, 1)
            self.update_video()

    def update_video(self):
        if self.cap is None or not self.cap.isOpened(): return
        ret, frame = self.cap.read()
        if ret:
            display_frame = frame.copy()
            cropped = frame[CROP_Y:CROP_Y+CROP_H, CROP_X:CROP_X+CROP_W]
            hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
            
            # 1. 모든 색상 통합 마스크
            combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for color_key, cfg in COLOR_RANGES.items():
                mask = cv2.inRange(hsv, np.array(cfg["low"]), np.array(cfg["high"]))
                combined_mask = cv2.bitwise_or(combined_mask, mask)
            
            combined_mask = cv2.dilate(cv2.erode(combined_mask, None, iterations=2), None, iterations=2)
            contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # 임시 리스트: (로봇좌표_x, 로봇좌표_y, 면적, 화면좌표_x, 화면좌표_y, 반지름)
            temp_targets = []
            
            for c in contours:
                area = cv2.contourArea(c) # 컨투어 면적 계산
                if area > 500: # 최소 면적 기준 (노이즈 제거)
                    ((x, y), radius) = cv2.minEnclosingCircle(c)
                    full_cx = int(x) + CROP_X
                    full_cy = int(y) + CROP_Y
                    
                    cx_raw = (full_cx - 640) * (500 / 900)
                    cy_raw = (full_cy - 360) * (500 / 900)
                    
                    if H_matrix is not None:
                        input_pt = np.array([[[cx_raw, cy_raw]]], dtype=np.float32)
                        robot_pt = cv2.perspectiveTransform(input_pt, H_matrix)
                        rx, ry = robot_pt[0][0][0], robot_pt[0][0][1]
                        
                        temp_targets.append({
                            'pos': [rx, ry],
                            'area': area,
                            'ui_pos': (full_cx, full_cy),
                            'r': int(radius)
                        })

            # 2. 면적(area) 기준으로 내림차순 정렬 (가장 큰 것이 0번 인덱스)
            temp_targets.sort(key=lambda x: x['area'], reverse=True)
            
            # 정제된 로봇 좌표 리스트 업데이트
            self.all_detected_targets = [t['pos'] for t in temp_targets]
            
            # 3. 화면 표시 (가장 큰 것만 강조하거나 순번 표시)
            for i, target in enumerate(temp_targets):
                color = (0, 255, 0) if i == 0 else (255, 255, 255) # 1순위는 녹색, 나머지는 흰색
                thickness = 3 if i == 0 else 1
                
                cv2.circle(display_frame, target['ui_pos'], target['r'], color, thickness)
                cv2.putText(display_frame, f"#{i+1} Area:{int(target['area'])}", 
                            (target['ui_pos'][0]-10, target['ui_pos'][1]-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # 자동 모드에서 참조할 타겟 (가장 큰 물체)
            if self.all_detected_targets:
                self.last_target_pos = self.all_detected_targets[0]
            else:
                self.last_target_pos = None

            # UI 영상 업데이트 로직 (동일)
            cv2.rectangle(display_frame, (CROP_X, CROP_Y), (CROP_X+CROP_W, CROP_Y+CROP_H), (255, 255, 255), 2)
            img = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB); img_pil = Image.fromarray(img)
            img_tk = ctk.CTkImage(light_image=img_pil, dark_image=img_pil, size=(900, 500))
            self.video_label.configure(image=img_tk)
            
        self.after(20, self.update_video)

    def start_socket_server(self):
        def server_thread():
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("0.0.0.0", 30002)); s.listen(1)
            while True:
                conn, addr = s.accept(); self.conn = conn
                self.status_label.configure(text=f"✅ Robot Connected", text_color="#1ABC9C")
        threading.Thread(target=server_thread, daemon=True).start()

    def send_command_and_wait(self, cmd):
        if not self.conn: return False
        try:
            self.conn.sendall((cmd + "\n").encode())
            self.conn.settimeout(25.0)
            return "DONE" in self.conn.recv(1024).decode()
        except: return False

    def _move_worker(self, mode, target_pos_at_click):
        if self.is_moving: return
        self.is_moving = True

        bx, by = None, None
        if target_pos_at_click:
            bx = np.clip(target_pos_at_click[0], X_MIN, X_MAX)
            by = np.clip(target_pos_at_click[1], Y_MIN, Y_MAX)

        if mode in ["STRIKE", "DUAL_STEP"]:
            if self.current_assigned_pos:
                bx, by = self.current_assigned_pos
            elif target_pos_at_click:
                bx = np.clip(target_pos_at_click[0], X_MIN, X_MAX)
                by = np.clip(target_pos_at_click[1], Y_MIN, Y_MAX)
        elif target_pos_at_click:
            bx = np.clip(target_pos_at_click[0], X_MIN, X_MAX)
            by = np.clip(target_pos_at_click[1], Y_MIN, Y_MAX)
            self.current_assigned_pos = (bx, by)

        # [동작 모드별 분기]
        if mode == "HOME":
            self.status_label.configure(text="🏠 Moving Home...", text_color="#3498db")
            if self.send_command_and_wait(f"MOVE,{HOME_POSE}"):
                self.current_assigned_pos = None

        elif mode == "SAFE":
            self.status_label.configure(text="🛡 Moving to Safe Pose...", text_color="#27ae60")
            if self.send_command_and_wait(f"MOVE,{SAFE_POSE}"):
                self.current_assigned_pos = None

        elif mode == "DISCHARGE":
            self.status_label.configure(text="📤 Discharging...", text_color="#8e44ad")
            if self.send_command_and_wait(f"MOVE,{DISCHARGE_POSE}"):
                self.current_assigned_pos = None
        elif mode == "ddischarge":
            self.status_label.configure(text="📤 Discharging...", text_color="#8e44ad")
            if self.send_command_and_wait(f"ddischarge"):
                self.current_assigned_pos = None
        elif mode == "badeinput":
            self.status_label.configure(text="📤 Discharging...", text_color="#8e44ad")
            if self.send_command_and_wait(f"badeinput"):
                self.current_assigned_pos = None
        elif mode == "frybade":
            self.status_label.configure(text="📤 Discharging...", text_color="#8e44ad")
            if self.send_command_and_wait(f"frybade"):
                self.current_assigned_pos = None
        elif mode == "fryoutput":
            self.status_label.configure(text="📤 Discharging...", text_color="#8e44ad")
            if self.send_command_and_wait(f"fryoutput"):
                self.current_assigned_pos = None

        elif mode == "DISCHARGE_SEQ":
            self.status_label.configure(text="🚀 Running Discharge Seq...", text_color="#2ecc71")
            discharge_path = [
                "670,758,1090,78,85,92",
                "975,12,1163.9,18.34,89.07,97.27",
                "999,-372,777,2.24,77.20,175.45",
                "943,30,1164,43.3,84,101.8"
            ]
            for i, pos in enumerate(discharge_path):
                print(f"--- Step {i+1} Start ---")
                if self.send_command_and_wait(f"MOVE,{pos}"):
                    print(f"✅ Step {i+1} 완료")
                    time.sleep(0.5)
                    if i == 2: time.sleep(2)
                else:
                    print(f"❌ Step {i+1} 실패"); break
            self.current_assigned_pos = None

        # [추가] SCREW 동작: 요청하신 4지점 3바퀴 반복
        elif mode == "SCREW":
            self.status_label.configure(text="🌀 Screwing (3 Rounds)...", text_color="#1ABC9C")
            
            # 속도와 블렌딩 값 설정 (여기서 조절 가능)
            screw_vel = 650   # mm/s
            screw_res = 300    # blending radius (mm) - 값이 클수록 모서리가 둥글어짐
            
            screw_points = [
                "1006,698,832",
                "745,1029,832",
                "613,781,871",
                "783,487,879"
            ]
            fixed_rot = "96,87,100"
            for r in range(3):
                for i, pt in enumerate(screw_points):
                    # 명령문에 속도와 블렌딩 추가
                    cmd = f"MOVE,{pt},{fixed_rot},{screw_vel},{screw_res}"
                    if not self.send_command_and_wait(cmd):
                        break
                    time.sleep(0.5)

        elif mode == "STRIKE" and bx is not None:
            self.status_label.configure(text="🔨 Fast Triple Striking...", text_color="#d35400")
            
            # 추천 속도: 500~800 (너무 빠르면 로봇이 흔들리니 테스트하며 조절)
            strike_vel = 1000
            strike_r = 3 # 털기 동작이므로 블렌딩은 0으로 해서 딱딱 끊어줘야 함
            
            for i in range(4):
                # 아래로 내리는 동작 (타격/털기)
                self.send_command_and_wait(f"MOVE,{bx:.2f},{by:.2f},877.00,{FIXED_ORIENTATION},{strike_vel},{strike_r}")
                # 위로 올리는 동작
                self.send_command_and_wait(f"MOVE,{bx:.2f},{by:.2f},950.00,{FIXED_ORIENTATION},{strike_vel},{strike_r}")

        elif mode == "TRACK" and bx is not None:
            offset_x = bx + 169
            offset_y = by + 150
            self.status_label.configure(text="🚀 Moving to Object...", text_color="#27ae60")
            self.send_command_and_wait(f"MOVE,{bx:.2f},{by:.2f},{MOVE_Z_DEPTH:.2f},{FIXED_ORIENTATION}")

        elif mode == "DUAL_STEP" and bx is not None:
            if bx < 670 and by > 758: selected_area = AREA_A; area_name = "Area A"
            elif bx < 670 and by <= 758: selected_area = AREA_B; area_name = "Area B"
            else: selected_area = AREA_C; area_name = "Area C"
            
            self.status_label.configure(text=f"🥄 Scoop: {area_name}", text_color="#c0392b")
            
            # 모든 구간에 동일하게 적용할 속도와 블렌딩 값
            scoop_vel = 100
            scoop_r = 10
            
            for i, step in enumerate(selected_area):
                tx, ty, tz = bx + step['x'], by + step['y'], MOVE_Z_DEPTH + step['z']
                
                # 모든 스텝에 scoop_r(50)을 그대로 적용
                cmd = f"MOVE,{tx:.2f},{ty:.2f},{tz:.2f},{step['rx']:.2f},{step['ry']:.2f},{step['rz']:.2f},{scoop_vel},{scoop_r}"
                
                print(f"DEBUG: Scoop Step {i+1} -> {cmd}")
                
                if not self.send_command_and_wait(cmd): 
                    break
            
            self.current_assigned_pos = None
        
        self.status_label.configure(text="✅ Complete", text_color="#1ABC9C")
        self.is_moving = False
    def toggle_all_auto(self):
        if not self.is_all_auto:
            self.is_all_auto = True
            self.stop_requested = False
            self.all_auto_btn.configure(text="🛑 STOP AFTER SESSION", fg_color="#c0392b")
            threading.Thread(target=self._all_auto_worker, daemon=True).start()
        else:
            self.stop_requested = True
            self.status_label.configure(text="⏳ Stopping after this session...", text_color="#f1c40f")

    def _all_auto_worker(self):
        while self.is_all_auto:
            try:
                # 1. 탐지된 물체 리스트 확인
                targets = list(self.all_detected_targets) if hasattr(self, 'all_detected_targets') else []
                
                if not targets:
                    self.status_label.configure(text="🔍 Searching for ANY object...", text_color="#3498db")
                    time.sleep(0.5)
                    continue

                # 2. 가장 가까운(또는 리스트의 첫 번째) 물체 선택
                target_pos = targets[0] 
                bx = np.clip(target_pos[0], X_MIN, X_MAX)
                by = np.clip(target_pos[1], Y_MIN, Y_MAX)

                # 3. 작업 속도 향상을 위해 blending과 속도 최적화
                # 트래킹 이동
                self.status_label.configure(text="🚀 Targeting...", text_color="#2ecc71")
                if not self.send_command_and_wait(f"MOVE,{bx:.2f},{by:.2f},{MOVE_Z_DEPTH:.2f},{FIXED_ORIENTATION},600,50"): break

                # 4. 스쿱(건져내기) 실행
                self._execute_scoop_logic(bx, by)

                # 5. 배출 시퀀스 (속도 대폭 상향 및 정지 최소화)
                self.status_label.configure(text="📤 Fast Discharge", text_color="#8e44ad")
                discharge_path = [
                    "670,758,1090,78,85,92,400,50",
                    "975,12,1163.9,18.34,89.07,97.27,400,50",
                    "999,-372,777,2.24,77.20,175.45", # 배출 지점은 정지
                    "943,30,1164,43.3,84,101.8,400,0"
                ]
                for pos in discharge_path:
                    self.send_command_and_wait(f"MOVE,{pos}")
                
                # 6. 세션 종료 체크
                if self.stop_requested:
                    break

            except Exception as e:
                print(f"Auto Error: {e}")
                break
        
        # 루프 종료 시 안전 위치 복귀 및 버튼 초기화
        self.send_command_and_wait(f"MOVE,{SAFE_POSE}")
        self.is_all_auto = False
        self.all_auto_btn.configure(text="🤖 ALL-AUTO START", fg_color="#e74c3c")

    def _execute_scoop_logic(self, bx, by):
        # 구역 판별
        if bx < 670 and by > 758: selected_area = AREA_A
        elif bx < 670 and by <= 758: selected_area = AREA_B
        else: selected_area = AREA_C
        
        scoop_vel = 250
        scoop_r = 50
        for step in selected_area:
            tx, ty, tz = bx + step['x'], by + step['y'], MOVE_Z_DEPTH + step['z']
            cmd = f"MOVE,{tx:.2f},{ty:.2f},{tz:.2f},{step['rx']:.2f},{step['ry']:.2f},{step['rz']:.2f},{scoop_vel},{scoop_r}"
            if not self.send_command_and_wait(cmd): break
    # 실행 함수들
    def execute_home_move(self): threading.Thread(target=self._move_worker, args=("HOME", None), daemon=True).start()
    def execute_safe_move(self): threading.Thread(target=self._move_worker, args=("SAFE", None), daemon=True).start()
    def execute_discharge_move(self): threading.Thread(target=self._move_worker, args=("DISCHARGE", None), daemon=True).start()
    def execute_discharge_sequence(self): threading.Thread(target=self._move_worker, args=("DISCHARGE_SEQ", None), daemon=True).start()
    def execute_screw_move(self): threading.Thread(target=self._move_worker, args=("SCREW", None), daemon=True).start()
    def execute_sub_program_call(self): threading.Thread(target=self._move_worker, args=("ddischarge", None), daemon=True).start()
    def execute_strike_move(self): 
        pos = list(self.last_target_pos) if self.last_target_pos else None
        threading.Thread(target=self._move_worker, args=("STRIKE", pos), daemon=True).start()
    def execute_marker_move(self): 
        pos = list(self.last_target_pos) if self.last_target_pos else None
        threading.Thread(target=self._move_worker, args=("TRACK", pos), daemon=True).start()
    def execute_dual_scoop(self): 
        pos = list(self.last_target_pos) if self.last_target_pos else None
        threading.Thread(target=self._move_worker, args=("DUAL_STEP", pos), daemon=True).start()

    def set_target_color(self, color_key): self.target_color_key = color_key
    def execute_named_sub_program(self, cmd_name):
        threading.Thread(target=self._move_worker, args=(cmd_name, None), daemon=True).start()

if __name__ == "__main__":
    ctk.set_appearance_mode("Dark")
    RobotApp().mainloop()