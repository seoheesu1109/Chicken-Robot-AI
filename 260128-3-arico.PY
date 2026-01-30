import socket
import tkinter as tk
import cv2
import numpy as np
import pyrealsense2 as rs
import threading
import collections
import time
from scipy.spatial.transform import Rotation as R

# 캘리브레이션 데이터
CAM_DATA = np.array([[304.6, -156.4, 984.6, -137.2, -16.3, -110.5], [-219.4, -248.5, 1310.1, -164.6, -4.5, -69.9], [372.4, -352.2, 1022.9, -141.0, -28.8, -148.3], [158.1, -247.5, 1015.5, -174.2, -21.6, -10.7], [-70.5, -455.2, 1600.2, -159.3, -49.1, -5.7], [-121.4, 173.4, 1984.4, -153.0, 23.0, 8.0], [404.8, 245.4, 1067.0, -169.0, -53.9, 2.6], [421.9, -304.0, 940.4, 132.0, -1.8, 20.3], [-101.9, -111.4, 423.7, 121.6, 5.0, 10.9]])
ROBOT_DATA = np.array([[971.42, -117.44, 969.11, 99.69, -47.15, 151.07], [676.39, 407.80, 848.89, 61.61, -80.63, 138.38], [1026.19, -4.95, 1159.95, 47.45, -22.50, -144.85], [830.43, -11.81, 981.49, 74.05, -68.44, -178.65], [1056.06, 553.84, 899.97, 96.65, -90.84, 166.60], [1039.0, 638.0, 245.0, 76.41, -86.71, 160.39], [1012.93, -284.34, 500.93, 101.28, -86.71, 160.39], [1012.93, -134.93, 1153.05, 85.05, -24.30, 134.89], [319.43, -198.54, 1153.05, 77.15, -9.18, 143.07]])

class ArUcoRobotMaster:
    def __init__(self, root):
        # 1. 속성 정의 (순서가 중요합니다!)
        self.root = root
        self.conn = None
        self.is_tracking = False 
        
        # 2. 필터 및 데드존 설정
        self.pos_history = collections.deque(maxlen=3)
        self.last_sent_pos = np.array([0.0, 0.0, 0.0])
        self.threshold = 10.0  # 10mm 이상 움직일 때만 전송 10이 적당함

        # 3. 캘리브레이션 및 카메라 초기화
        self.R_base, self.T_base = self.calibrate_base(CAM_DATA, ROBOT_DATA)
        self.setup_camera()
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters()

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
        profile = self.pipeline.start(config)
        intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.camera_matrix = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]])
        self.dist_coeffs = np.array(intr.coeffs)

    def camera_worker(self):
        while not self.stop_event.is_set():
            frames = self.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame: continue

            img = np.asanyarray(color_frame.get_data())
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)

            if ids is not None:
                cv2.aruco.drawDetectedMarkers(img, corners, ids)
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 100, self.camera_matrix, self.dist_coeffs)
                marker_map = {ids[i][0]: i for i in range(len(ids))}
                
                if 1 in marker_map:
                    idx = marker_map[1]
                    # 현재 좌표 계산
                    pos = np.dot(self.R_base, tvecs[idx][0]) + self.T_base
                    pos[0] -= 22.57; pos[1] += 41.75; pos[2] += -100.45

                    # 노이즈 필터링
                    self.pos_history.append(pos)
                    smooth_pos = np.mean(self.pos_history, axis=0)

                    # --- [데드존 필터 로직] ---
                    dist = np.linalg.norm(smooth_pos - self.last_sent_pos)
                    
                    if self.is_tracking and self.conn:
                        if dist > self.threshold: # 변화량이 threshold보다 클 때만 전송
                            try:
                                f_rx, f_ry, f_rz = 0.0, 180.0, 0.0
                                msg = f"MOVE,{smooth_pos[0]:.1f},{smooth_pos[1]:.1f},{smooth_pos[2]:.1f},{f_rx},{f_ry},{f_rz}\n"
                                self.conn.sendall(msg.encode())
                                self.last_sent_pos = smooth_pos # 마지막 보낸 좌표 갱신
                            except: pass

                    # 마커 시각화 표시
                    c_x, c_y = int(corners[idx][0][:, 0].mean()), int(corners[idx][0][:, 1].mean())
                    cv2.circle(img, (c_x, c_y), 7, (0, 0, 255), -1)
                    cv2.putText(img, f"ID:1 DETECTED", (c_x - 40, c_y - 30), 1, 1.2, (0, 255, 0), 2)
            else:
                cv2.putText(img, "SEARCHING FOR ID:1...", (50, 60), 1, 2.0, (0, 0, 255), 2)

            cv2.imshow("Zero-Delay Smart Tracker", cv2.resize(img, (960, 540)))
            if cv2.waitKey(1) == ord('q'): break

    def setup_ui(self):
        tk.Label(self.root, text="[ Zero-Delay Smart Tracker ]", font=("Arial", 14, "bold")).pack(pady=20)
        self.track_btn = tk.Button(self.root, text="START TRACKING", bg="red", fg="white", 
                                   font=("Arial", 12, "bold"), height=2, command=self.toggle_tracking)
        self.track_btn.pack(fill="x", padx=50, pady=20)
        self.status = tk.Label(self.root, text="연결 대기 중...", bd=1, relief="sunken", anchor="w")
        self.status.pack(side="bottom", fill="x")

    def toggle_tracking(self):
        self.is_tracking = not self.is_tracking
        self.track_btn.config(text="STOP" if self.is_tracking else "START TRACKING", bg="black" if self.is_tracking else "red")

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
    app = ArUcoRobotMaster(root)
    root.mainloop()