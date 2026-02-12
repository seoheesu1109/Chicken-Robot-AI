import socket
import tkinter as tk
import cv2
import numpy as np
import pyrealsense2 as rs
import threading
import collections
import time
from ultralytics import YOLO, YOLOWorld

# [1] 데이터 및 좌표 설정
CAM_DATA = np.array([[304.6, -156.4, 984.6, -137.2, -16.3, -110.5], [-219.4, -248.5, 1310.1, -164.6, -4.5, -69.9], [372.4, -352.2, 1022.9, -141.0, -28.8, -148.3], [158.1, -247.5, 1015.5, -174.2, -21.6, -10.7], [-70.5, -455.2, 1600.2, -159.3, -49.1, -5.7], [-121.4, 173.4, 1984.4, -153.0, 23.0, 8.0], [404.8, 245.4, 1067.0, -169.0, -53.9, 2.6], [421.9, -304.0, 940.4, 132.0, -1.8, 20.3], [-101.9, -111.4, 423.7, 121.6, 5.0, 10.9]])
ROBOT_DATA = np.array([[971.42, -117.44, 969.11, 99.69, -47.15, 151.07], [676.39, 407.80, 848.89, 61.61, -80.63, 138.38], [1026.19, -4.95, 1159.95, 47.45, -22.50, -144.85], [830.43, -11.81, 981.49, 74.05, -68.44, -178.65], [1056.06, 553.84, 899.97, 96.65, -90.84, 166.60], [1039.0, 638.0, 245.0, 76.41, -86.71, 160.39], [1012.93, -284.34, 500.93, 101.28, -86.71, 160.39], [1012.93, -134.93, 1153.05, 85.05, -24.30, 134.89], [319.43, -198.54, 1153.05, 77.15, -9.18, 143.07]])

# 안전 작업 범위 (Space Limit)
X_MIN_LIMIT, X_MAX_LIMIT = 366.0, 1600.0 
Y_MIN_LIMIT, Y_MAX_LIMIT = -484.62, 900.0
Z_MIN_LIMIT, Z_MAX_LIMIT = 60.0, 1018.42

# 벽(Box) 장애물 및 배출 지점
WALL_X, WALL_Y = (508.88, 1489.08), (59.99, 572.37)
SAFE_Z_OVER = 450.0
DROP_POS = [1460.0, -335.0, 122.0]

# [추가] 사람 감지 안전 구역 (테스트 현장 상황에 맞춰 수정하세요)
# 예: 로봇 정면의 특정 좌표 범위
SAFE_ZONE_X = (366.0, 1600.0) 
SAFE_ZONE_Y = (-484.0, 900.0)

class ChickenRobotMaster:
    def __init__(self, root):
        self.root = root
        self.conn = None
        self.is_tracking = False 
        self.is_waiting = False
        self.is_emergency = False # 비상 정지 상태 플래그

        # [모델 설정] 치킨용 YOLOv8 + 안전용 YOLO-World
        self.model = YOLO(r"C:\runs\detect\train12\weights\best.pt")
        self.safety_model = YOLOWorld('yolov8s-worldv2.pt') 
        self.safety_model.set_classes(["person"]) # 사람만 검출하도록 최적화

        self.pos_history = collections.deque(maxlen=5) 
        self.last_sent_pos = np.array([400.0, 0.0, SAFE_Z_OVER]) 

        self.R_base, self.T_base = self.calibrate_base(CAM_DATA, ROBOT_DATA)
        self.setup_camera()
        self.setup_ui()

        self.stop_event = threading.Event()
        self.cam_thread = threading.Thread(target=self.camera_worker, daemon=True)
        self.cam_thread.start()
        self.start_server()

    def calibrate_base(self, cam_pts, robot_pts):
        A, B = cam_pts[:, :3], robot_pts[:, :3]
        cA, cB = np.mean(A, axis=0), np.mean(B, axis=0)
        H = np.dot((A - cA).T, (B - cB))
        U, S, Vt = np.linalg.svd(H)
        R_mat = np.dot(Vt.T, U.T)
        if np.linalg.det(R_mat) < 0: Vt[2,:] *= -1; R_mat = np.dot(Vt.T, U.T)
        return R_mat, cB - np.dot(R_mat, cA)

    def setup_camera(self):
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        profile = self.pipeline.start(config)
        self.align = rs.align(rs.stream.color)
        self.intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

    def send_robot_command(self, x, y, z):
        if self.conn and not self.is_emergency:
            try:
                x = max(min(x, X_MAX_LIMIT), X_MIN_LIMIT)
                y = max(min(y, Y_MAX_LIMIT), Y_MIN_LIMIT)
                z = max(min(z, Z_MAX_LIMIT), Z_MIN_LIMIT)
                msg = f"MOVE,{x:.1f},{y:.1f},{z:.1f},0.0,180.0,0.0\n"
                self.conn.sendall(msg.encode())
                time.sleep(0.05)
            except: pass

    def send_robot_stop(self):
        """비상 정지 명령 전송"""
        if self.conn:
            try:
                msg = "STOP\n"
                self.conn.sendall(msg.encode())
                print("🚨 [SENT] Emergency STOP command to Robot")
            except: pass

    def camera_worker(self):
        while not self.stop_event.is_set():
            frames = self.pipeline.wait_for_frames()
            aligned = self.align.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if not color_frame or not depth_frame: continue

            img = np.asanyarray(color_frame.get_data())
            
            # [1] 안전 검사 (YOLO-World)
            safety_results = self.safety_model.predict(img, conf=0.4, verbose=False)
            person_in_zone = False
            for r in safety_results:
                for box in r.boxes:
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    u, v = int((xyxy[0]+xyxy[2])/2), int((xyxy[1]+xyxy[3])/2)
                    depth = depth_frame.get_distance(u, v)
                    if 0.1 < depth < 3.0:
                        pt_cam = np.array(rs.rs2_deproject_pixel_to_point(self.intrinsics, [u, v], depth)) * 1000.0
                        p_pos = np.dot(self.R_base, pt_cam) + self.T_base
                        
                        # 안전 구역 침범 확인
                        if (SAFE_ZONE_X[0] < p_pos[0] < SAFE_ZONE_X[1]) and (SAFE_ZONE_Y[0] < p_pos[1] < SAFE_ZONE_Y[1]):
                            person_in_zone = True
                            cv2.rectangle(img, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 0, 255), 3)
                            cv2.putText(img, "DANGER: PERSON", (xyxy[0], xyxy[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

            if person_in_zone:
                if not self.is_emergency:
                    self.is_emergency = True
                    self.is_tracking = False
                    self.send_robot_stop()
                    self.update_ui_emergency(True)
            else:
                if self.is_emergency: # 사람이 구역을 벗어났을 때 자동 해제 (원치 않으면 수동으로 변경 가능)
                    self.is_emergency = False
                    self.update_ui_emergency(False)

            # [2] 치킨 검출 (YOLOv8) - 비상 상황이 아닐 때만 작동
            if not self.is_emergency:
                results = self.model.predict(img, conf=0.7, imgsz=1024, verbose=False)
                for r in results:
                    for box in r.boxes:
                        xyxy = box.xyxy[0].cpu().numpy().astype(int)
                        u, v = int((xyxy[0]+xyxy[2])/2), int((xyxy[1]+xyxy[3])/2)
                        depth = depth_frame.get_distance(u, v)
                        if 0.1 < depth < 2.0:
                            pt_cam = np.array(rs.rs2_deproject_pixel_to_point(self.intrinsics, [u, v], depth)) * 1000.0
                            pos = np.dot(self.R_base, pt_cam) + self.T_base
                            pos[0] -= 22.57; pos[1] += 41.75; pos[2] -= 100.45 
                            
                            self.pos_history.append(pos)
                            smooth_pos = np.mean(self.pos_history, axis=0)

                            if self.is_tracking and not self.is_waiting:
                                dist = np.linalg.norm(smooth_pos - self.last_sent_pos)
                                if dist < 5.0:
                                    self.is_waiting = True
                                    threading.Timer(2.0, self.auto_drop_sequence, args=[smooth_pos]).start()
                                    break
                                elif dist > 15.0:
                                    self.send_robot_command(smooth_pos[0], smooth_pos[1], smooth_pos[2])
                                    self.last_sent_pos = smooth_pos.copy()

            cv2.imshow("Chicken Smart Safety System", cv2.resize(img, (1280, 720)))
            if cv2.waitKey(1) == ord('q'): break

    def update_ui_emergency(self, is_danger):
        if is_danger:
            self.track_btn.config(text="🚨 EMERGENCY STOP 🚨", bg="orange")
            self.status.config(text="위험: 안전 구역 내 사람 감지!", fg="red")
        else:
            self.track_btn.config(text="START TRACKING", bg="red")
            self.status.config(text="안전: 작업 가능 상태", fg="blue")

    def auto_drop_sequence(self, current_pos):
        if self.is_emergency: return
        self.send_robot_command(current_pos[0], current_pos[1], SAFE_Z_OVER)
        time.sleep(1.2)
        self.send_robot_command(DROP_POS[0], DROP_POS[1], SAFE_Z_OVER)
        time.sleep(3.0) 
        self.send_robot_command(DROP_POS[0], DROP_POS[1], DROP_POS[2])
        time.sleep(2.0)
        self.send_robot_command(DROP_POS[0], DROP_POS[1], SAFE_Z_OVER)
        time.sleep(1.2)
        self.is_waiting = False

    def setup_ui(self):
        tk.Label(self.root, text="[ Chicken Robot Master v2.1 (Safety) ]", font=("Arial", 14, "bold")).pack(pady=20)
        self.track_btn = tk.Button(self.root, text="START TRACKING", bg="red", fg="white", 
                                   font=("Arial", 12, "bold"), height=2, command=self.toggle_tracking)
        self.track_btn.pack(fill="x", padx=50, pady=20)
        self.status = tk.Label(self.root, text="연결 대기 중...", bd=1, relief="sunken", anchor="w")
        self.status.pack(side="bottom", fill="x")

    def toggle_tracking(self):
        if self.is_emergency: return
        self.is_tracking = not self.is_tracking
        self.track_btn.config(text="STOP" if self.is_tracking else "START TRACKING", 
                              bg="black" if self.is_tracking else "red")

    def start_server(self):
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind(("0.0.0.0", 30002)); self.server.listen(1)
        self.root.after(100, self.accept_conn)

    def accept_conn(self):
        self.server.setblocking(False)
        try:
            self.conn, addr = self.server.accept()
            self.status.config(text=f"CONNECTED: {addr}", fg="green")
        except: self.root.after(500, self.accept_conn)

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Chicken Robot Master v2.1")
    app = ChickenRobotMaster(root)
    root.mainloop()