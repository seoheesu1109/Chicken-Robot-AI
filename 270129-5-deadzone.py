import socket
import tkinter as tk
import cv2
import numpy as np
import pyrealsense2 as rs
import threading
import collections
import time
from ultralytics import YOLO

# [1] 데이터 및 좌표 설정
CAM_DATA = np.array([[304.6, -156.4, 984.6, -137.2, -16.3, -110.5], [-219.4, -248.5, 1310.1, -164.6, -4.5, -69.9], [372.4, -352.2, 1022.9, -141.0, -28.8, -148.3], [158.1, -247.5, 1015.5, -174.2, -21.6, -10.7], [-70.5, -455.2, 1600.2, -159.3, -49.1, -5.7], [-121.4, 173.4, 1984.4, -153.0, 23.0, 8.0], [404.8, 245.4, 1067.0, -169.0, -53.9, 2.6], [421.9, -304.0, 940.4, 132.0, -1.8, 20.3], [-101.9, -111.4, 423.7, 121.6, 5.0, 10.9]])
ROBOT_DATA = np.array([[971.42, -117.44, 969.11, 99.69, -47.15, 151.07], [676.39, 407.80, 848.89, 61.61, -80.63, 138.38], [1026.19, -4.95, 1159.95, 47.45, -22.50, -144.85], [830.43, -11.81, 981.49, 74.05, -68.44, -178.65], [1056.06, 553.84, 899.97, 96.65, -90.84, 166.60], [1039.0, 638.0, 245.0, 76.41, -86.71, 160.39], [1012.93, -284.34, 500.93, 101.28, -86.71, 160.39], [1012.93, -134.93, 1153.05, 85.05, -24.30, 134.89], [319.43, -198.54, 1153.05, 77.15, -9.18, 143.07]])

# 안전 작업 범위 (Space Limit)
X_MIN_LIMIT, X_MAX_LIMIT = 366.0, 1600.0 
Y_MIN_LIMIT, Y_MAX_LIMIT = -484.62, 900.0
Z_MIN_LIMIT, Z_MAX_LIMIT = 60.0, 1018.42

# 벽(Box) 장애물 설정
WALL_X = (508.88, 1489.08)
WALL_Y = (59.99, 572.37)
WALL_Z_TOP = 319.54
SAFE_Z_OVER = 450.0  # 벽을 넘기 위한 안전 높이

# 배출 지점
DROP_POS = [1460.0, -335.0, 122.0]

class ChickenRobotMaster:
    def __init__(self, root):
        self.root = root
        self.conn = None
        self.is_tracking = False 
        self.is_waiting = False # 2초 대기 상태 플래그
        
        # 모델 및 히스토리 초기화
        self.model = YOLO(r"C:\runs\detect\train12\weights\best.pt")
        self.pos_history = collections.deque(maxlen=5) 
        self.last_sent_pos = np.array([400.0, 0.0, SAFE_Z_OVER]) 
        self.threshold = 15.0  

        # 캘리브레이션 및 카메라 세팅
        self.R_base, self.T_base = self.calibrate_base(CAM_DATA, ROBOT_DATA)
        self.setup_camera()
        self.setup_ui()

        # 스레드 시작
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
        """좌표 명령 전송"""
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
        """벽 평면 영역 내에 있는지 확인"""
        x_min, x_max = min(WALL_X), max(WALL_X)
        y_min, y_max = min(WALL_Y), max(WALL_Y)
        return (x_min <= x <= x_max) and (y_min <= y <= y_max)

    def execute_smart_move(self, target_pos):
        """벽 회피 지능형 이동"""
        in_wall_now = self.is_in_wall_plane(self.last_sent_pos[0], self.last_sent_pos[1])
        in_wall_dest = self.is_in_wall_plane(target_pos[0], target_pos[1])
        
        if in_wall_now or in_wall_dest:
            self.send_robot_command(self.last_sent_pos[0], self.last_sent_pos[1], SAFE_Z_OVER)
            self.send_robot_command(target_pos[0], target_pos[1], SAFE_Z_OVER)
            self.send_robot_command(target_pos[0], target_pos[1], target_pos[2])
        else:
            self.send_robot_command(target_pos[0], target_pos[1], target_pos[2])
        self.last_sent_pos = target_pos.copy()

    def camera_worker(self):
        while not self.stop_event.is_set():
            frames = self.pipeline.wait_for_frames()
            aligned = self.align.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if not color_frame or not depth_frame: continue

            img = np.asanyarray(color_frame.get_data())
            results = self.model.predict(img, conf=0.7, imgsz=1024, verbose=False)
            
            chicken_detected = False
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
                        chicken_detected = True

                        # 트래킹 중이고 대기 상태가 아닐 때만 이동
                        if self.is_tracking and self.conn and not self.is_waiting:
                            dist_to_target = np.linalg.norm(smooth_pos - self.last_sent_pos)
                            
                            # 치킨 도달 판정 (5mm 이내)
                            if dist_to_target < 5.0:
                                print("🎯 치킨 도달! 2초 대기 후 배출 시퀀스 가동...")
                                self.is_waiting = True
                                threading.Timer(2.0, self.auto_drop_sequence, args=[smooth_pos]).start()
                                break

                            if dist_to_target > self.threshold:
                                self.execute_smart_move(smooth_pos)

            cv2.imshow("Chicken Smart Tracker", cv2.resize(img, (1280, 720)))
            if cv2.waitKey(1) == ord('q'): break

    def auto_drop_sequence(self, current_pos):
        """2초 대기 후 벽 넘어 배출 (Z축 하강 동작 보장 버전)"""
        print("🚀 [1/3] 배출 시퀀스 시작: 안전 높이로 상승")
        
        # 1. 현재 위치에서 수직 상승 (450까지)
        self.send_robot_command(current_pos[0], current_pos[1], SAFE_Z_OVER)
        time.sleep(1.2) # 위로 올라갈 시간

        print(f"🚀 [2/3] 벽 넘어 이동 중: {DROP_POS[0]}, {DROP_POS[1]}")
        # 2. 벽을 가로질러 목표 지점 상공(450)으로 이동
        self.send_robot_command(DROP_POS[0], DROP_POS[1], SAFE_Z_OVER)
        
        # 중요: 이동 거리가 멀기 때문에 도달할 때까지 충분히 기다려야 합니다.
        # 로봇 속도에 따라 2.5 ~ 3.5초 정도가 적당합니다.
        time.sleep(3.0) 

        print(f"📉 [3/3] 최종 하강: Z={DROP_POS[2]}")
        # 3. 목표 지점에 도착한 '후'에 하강 명령 전송
        self.send_robot_command(DROP_POS[0], DROP_POS[1], DROP_POS[2])
        time.sleep(2.0) # 내려가서 배출(그리퍼 작동 등)할 시간

        # 4. 배출 완료 후 복귀
        print("✅ 배출 완료. 다시 안전 높이로 복귀합니다.")
        self.send_robot_command(DROP_POS[0], DROP_POS[1], SAFE_Z_OVER)
        time.sleep(1.2)
        
        # 상태 초기화
        self.last_sent_pos = np.array([DROP_POS[0], DROP_POS[1], SAFE_Z_OVER])
        self.is_waiting = False
        print("🔄 검색 모드로 복귀 완료.")

    def setup_ui(self):
        tk.Label(self.root, text="[ YOLOv8 Chicken Smart Tracker ]", font=("Arial", 14, "bold")).pack(pady=20)
        self.track_btn = tk.Button(self.root, text="START TRACKING", bg="red", fg="white", 
                                   font=("Arial", 12, "bold"), height=2, command=self.toggle_tracking)
        self.track_btn.pack(fill="x", padx=50, pady=20)
        self.status = tk.Label(self.root, text="연결 대기 중...", bd=1, relief="sunken", anchor="w")
        self.status.pack(side="bottom", fill="x")

    def toggle_tracking(self):
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
    root.title("Chicken Robot Master v2.0")
    app = ChickenRobotMaster(root)
    root.mainloop()