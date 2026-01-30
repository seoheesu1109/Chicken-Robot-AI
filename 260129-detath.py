import socket
import tkinter as tk
import cv2
import numpy as np
import pyrealsense2 as rs
import threading
import collections
from ultralytics import YOLO

# 캘리브레이션 데이터 (기존 데이터 유지)
CAM_DATA = np.array([[304.6, -156.4, 984.6, -137.2, -16.3, -110.5], [-219.4, -248.5, 1310.1, -164.6, -4.5, -69.9], [372.4, -352.2, 1022.9, -141.0, -28.8, -148.3], [158.1, -247.5, 1015.5, -174.2, -21.6, -10.7], [-70.5, -455.2, 1600.2, -159.3, -49.1, -5.7], [-121.4, 173.4, 1984.4, -153.0, 23.0, 8.0], [404.8, 245.4, 1067.0, -169.0, -53.9, 2.6], [421.9, -304.0, 940.4, 132.0, -1.8, 20.3], [-101.9, -111.4, 423.7, 121.6, 5.0, 10.9]])
ROBOT_DATA = np.array([[971.42, -117.44, 969.11, 99.69, -47.15, 151.07], [676.39, 407.80, 848.89, 61.61, -80.63, 138.38], [1026.19, -4.95, 1159.95, 47.45, -22.50, -144.85], [830.43, -11.81, 981.49, 74.05, -68.44, -178.65], [1056.06, 553.84, 899.97, 96.65, -90.84, 166.60], [1039.0, 638.0, 245.0, 76.41, -86.71, 160.39], [1012.93, -284.34, 500.93, 101.28, -86.71, 160.39], [1012.93, -134.93, 1153.05, 85.05, -24.30, 134.89], [319.43, -198.54, 1153.05, 77.15, -9.18, 143.07]])

class ChickenRobotMaster:
    def __init__(self, root):
        self.root = root
        self.conn = None
        self.is_tracking = False 
        
        # 1. YOLO 모델 로드 (경로 확인 필수)
        self.model = YOLO(r"C:\runs\detect\train12\weights\best.pt")
        
        # 2. 필터 및 데드존 설정
        self.pos_history = collections.deque(maxlen=5) # 필터링 강화
        self.last_sent_pos = np.array([0.0, 0.0, 0.0])
        self.threshold = 100.0  # 10mm 변동 시 전송

        # 3. 캘리브레이션 및 카메라 초기화
        self.R_base, self.T_base = self.calibrate_base(CAM_DATA, ROBOT_DATA)
        self.setup_camera()

        # 4. UI 및 통신 시작
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
        
        # Color와 Depth 좌표계를 맞추기 위한 Align 설정
        self.align = rs.align(rs.stream.color)
        
        intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.intrinsics = intr 

    def camera_worker(self):
        while not self.stop_event.is_set():
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)
            
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            if not color_frame or not depth_frame: continue

            img = np.asanyarray(color_frame.get_data())
            results = self.model.predict(img, conf=0.7, imgsz=1024, verbose=False)
            
            chicken_detected = False
            for r in results:
                for box in r.boxes:
                    # 1. 감지 좌표 추출
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    u, v = int((xyxy[0] + xyxy[2]) / 2), int((xyxy[1] + xyxy[3]) / 2)
                    
                    depth_val = depth_frame.get_distance(u, v)
                    
                    if 0.1 < depth_val < 2.0:
                        point_cam = rs.rs2_deproject_pixel_to_point(self.intrinsics, [u, v], depth_val)
                        point_cam = np.array(point_cam) * 1000.0
                        pos = np.dot(self.R_base, point_cam) + self.T_base
                        
                        # 오프셋 및 필터 적용
                        pos[0] -= 72.57; pos[1] += 81.75; pos[2] += -100.45 #pos[0] -= 22.57; pos[1] += 41.75; pos[2] += -100.45
                        self.pos_history.append(pos)
                        smooth_pos = np.mean(self.pos_history, axis=0)
                        chicken_detected = True

                        # --- [ 시각화 강화: 치킨 감지 표시 ] ---
                        # A. 바운딩 박스 (초록색)
                        cv2.rectangle(img, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)
                        
                        # B. 중앙 조준점 (Crosshair)
                        cv2.drawMarker(img, (u, v), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
                        
                        # C. 상단 상태 바 및 좌표 표시
                        label = f"CHICKEN DETECTED! [X:{smooth_pos[0]:.0f} Y:{smooth_pos[1]:.0f} Z:{smooth_pos[2]:.0f}]"
                        cv2.rectangle(img, (xyxy[0], xyxy[1]-35), (xyxy[0]+450, xyxy[1]), (0, 255, 0), -1) # 텍스트 배경
                        cv2.putText(img, label, (xyxy[0]+10, xyxy[1]-10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

                        # 로봇 전송 로직 (기존과 동일)
                        if self.is_tracking and self.conn:
                            dist = np.linalg.norm(smooth_pos - self.last_sent_pos)
                            if dist > self.threshold:
                                try:
                                    f_rx, f_ry, f_rz = 0.0, 180.0, 0.0
                                    msg = f"MOVE,{smooth_pos[0]:.1f},{smooth_pos[1]:.1f},{smooth_pos[2]:.1f},{f_rx},{f_ry},{f_rz}\n"
                                    self.conn.sendall(msg.encode())
                                    self.last_sent_pos = smooth_pos.copy()
                                except: pass

            # 감지되지 않았을 때 안내
            if not chicken_detected:
                cv2.rectangle(img, (30, 30), (450, 80), (0, 0, 255), -1)
                cv2.putText(img, "STATE: SEARCHING FOR CHICKEN...", (40, 65), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # 화면 출력
            cv2.imshow("Chicken Tracking System", cv2.resize(img, (1280, 720)))
            if cv2.waitKey(1) == ord('q'): break

    def setup_ui(self):
        tk.Label(self.root, text="[ YOLOv8 Chicken Tracker ]", font=("Arial", 14, "bold")).pack(pady=20)
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
    root.title("YOLOv8 Robot Control")
    app = ChickenRobotMaster(root)
    root.mainloop()