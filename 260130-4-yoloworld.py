import socket
import tkinter as tk
import cv2
import numpy as np
import threading
import collections
import time
from ultralytics import YOLOWorld

# [1] 데이터 및 좌표 설정 (웹캠용으로 조정)
# 웹캠은 Depth가 없으므로 캘리브레이션 대신 픽셀-좌표 변환 비율(Scale)을 임시로 사용합니다.
PIXEL_TO_MM = 1.5  # 1픽셀당 몇 mm인지 (테스트하며 조정 필요)
CENTER_U, CENTER_V = 320, 240 # 웹캠 해상도 640x480 기준 중심점

# 안전 범위 및 벽 설정
X_MIN_LIMIT, X_MAX_LIMIT = 366.0, 1600.0 
Y_MIN_LIMIT, Y_MAX_LIMIT = -484.62, 900.0
Z_MIN_LIMIT, Z_MAX_LIMIT = 60.0, 1018.42
WALL_X, WALL_Y = (508.88, 1489.08), (59.99, 572.37)
SAFE_Z_OVER = 450.0
DROP_POS = [1460.0, -335.0, 122.0]

class ChickenRobotMaster:
    def __init__(self, root):
        self.root = root
        self.conn = None
        self.is_tracking = False 
        self.is_waiting = False 
        
        # YOLO-World 모델 로드 (웹캠 테스트용이므로 제일 가벼운 's' 모델 권장)
        self.model = YOLOWorld('yolov8s-worldv2.pt') 
        self.menu_list = ["fried chicken", "spicy chicken", "potato chips"]
        self.model.set_classes(self.menu_list)

        self.pos_history = collections.deque(maxlen=5) 
        self.last_sent_pos = np.array([400.0, 0.0, SAFE_Z_OVER]) 
        self.threshold = 15.0  

        self.setup_ui()
        self.stop_event = threading.Event()
        
        # 웹캠 시작 (0번은 기본 내장 캠 또는 첫 번째 USB 캠)
        self.cap = cv2.VideoCapture(0)
        
        self.cam_thread = threading.Thread(target=self.camera_worker, daemon=True)
        self.cam_thread.start()
        self.start_server()

    def send_robot_command(self, x, y, z):
        if self.conn:
            try:
                x = max(min(x, X_MAX_LIMIT), X_MIN_LIMIT)
                y = max(min(y, Y_MAX_LIMIT), Y_MIN_LIMIT)
                z = max(min(z, Z_MAX_LIMIT), Z_MIN_LIMIT)
                msg = f"MOVE,{x:.1f},{y:.1f},{z:.1f},0.0,180.0,0.0\n"
                self.conn.sendall(msg.encode())
                time.sleep(0.05)
            except: pass

    def is_in_wall_plane(self, x, y):
        x_min, x_max = min(WALL_X), max(WALL_X)
        y_min, y_max = min(WALL_Y), max(WALL_Y)
        return (x_min <= x <= x_max) and (y_min <= y <= y_max)

    def execute_smart_move(self, target_pos):
        in_wall_now = self.is_in_wall_plane(self.last_sent_pos[0], self.last_sent_pos[1])
        in_wall_dest = self.is_in_wall_plane(target_pos[0], target_pos[1])
        if in_wall_now or in_wall_dest:
            self.send_robot_command(self.last_sent_pos[0], self.last_sent_pos[1], SAFE_Z_OVER)
            self.send_robot_command(target_pos[0], target_pos[1], SAFE_Z_OVER)
        self.send_robot_command(target_pos[0], target_pos[1], target_pos[2])
        self.last_sent_pos = target_pos.copy()

    def camera_worker(self):
        # [수정] 모델이 더 잘 찾을 수 있도록 단어 꾸러미를 풍성하게 만듭니다.
        # "item"이나 "object"를 넣으면 뭐라도 잡으려고 노력합니다.
        self.menu_list = ["chicken", "fried food", "nugget", "brown snack", "hand", "person"]
        self.model.set_classes(self.menu_list)

        while not self.stop_event.is_set():
            ret, frame = self.cap.read()
            if not ret: continue

            # [수정] conf를 0.1로 더 낮춰서 민감하게 반응하게 합니다.
            results = self.model.predict(frame, conf=0.1, imgsz=640, verbose=False)
            
            for r in results:
                for box in r.boxes:
                    conf_val = float(box.conf[0]) # 확신도 (0~1)
                    cls_id = int(box.cls[0])
                    label = self.menu_list[cls_id]
                    
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    
                    # 화면에 박스, 이름, 확신도를 모두 표시
                    # 만약 치킨을 비췄는데 'hand'라고 뜨면 단어를 조정해야 합니다.
                    color = (0, 255, 0) if conf_val > 0.2 else (0, 165, 255) # 확신도 낮으면 주황색
                    cv2.rectangle(frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color, 2)
                    cv2.putText(frame, f"{label} {conf_val:.2f}", (xyxy[0], xyxy[1]-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    # [트래킹 로직] - label이 person이나 hand가 아닐 때만 로봇 가동
                    if self.is_tracking and self.conn and not self.is_waiting:
                        if label not in ["person", "hand"]:
                            # ... (이전과 동일한 좌표 변환 및 전송 로직) ...
                            u, v = int((xyxy[0]+xyxy[2])/2), int((xyxy[1]+xyxy[3])/2)
                            pos_x = 800.0 + (v - CENTER_V) * PIXEL_TO_MM 
                            pos_y = 0.0 + (u - CENTER_U) * PIXEL_TO_MM
                            smooth_pos = np.array([pos_x, pos_y, 150.0])
                            
                            dist_to_target = np.linalg.norm(smooth_pos - self.last_sent_pos)
                            if dist_to_target > self.threshold:
                                self.execute_smart_move(smooth_pos)

            cv2.imshow("Webcam YOLO-World Test", frame)
            if cv2.waitKey(1) == ord('q'): break

    def auto_drop_sequence(self, current_pos):
        self.send_robot_command(current_pos[0], current_pos[1], SAFE_Z_OVER)
        time.sleep(1.0)
        self.send_robot_command(DROP_POS[0], DROP_POS[1], SAFE_Z_OVER)
        time.sleep(2.5)
        self.send_robot_command(DROP_POS[0], DROP_POS[1], DROP_POS[2])
        time.sleep(2.0)
        self.send_robot_command(DROP_POS[0], DROP_POS[1], SAFE_Z_OVER)
        self.is_waiting = False

    def setup_ui(self):
        tk.Label(self.root, text="[ Webcam YOLO-World Tracker ]", font=("Arial", 14, "bold")).pack(pady=20)
        self.track_btn = tk.Button(self.root, text="START TRACKING", bg="red", fg="white", font=("Arial", 12, "bold"), height=2, command=self.toggle_tracking)
        self.track_btn.pack(fill="x", padx=50, pady=20)
        self.status = tk.Label(self.root, text="연결 대기 중...", bd=1, relief="sunken", anchor="w")
        self.status.pack(side="bottom", fill="x")

    def toggle_tracking(self):
        self.is_tracking = not self.is_tracking
        self.track_btn.config(text="STOP" if self.is_tracking else "START TRACKING", bg="black" if self.is_tracking else "red")

    def start_server(self):
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM); self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind(("0.0.0.0", 30002)); self.server.listen(1); self.root.after(100, self.accept_conn)

    def accept_conn(self):
        self.server.setblocking(False)
        try: self.conn, addr = self.server.accept(); self.status.config(text=f"CONNECTED: {addr}", fg="green")
        except: self.root.after(500, self.accept_conn)

if __name__ == "__main__":
    root = tk.Tk(); app = ChickenRobotMaster(root); root.mainloop()