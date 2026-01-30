import socket
import tkinter as tk
from tkinter import messagebox
import cv2
import numpy as np
import pyrealsense2 as rs
import threading
from scipy.spatial.transform import Rotation as R

# --- [1. 캘리브레이션 데이터 (오늘 수정한 최종본)] ---
# 주신 9세트 데이터를 기반으로 미리 계산된 행렬을 사용하거나, 
# 프로그램 시작 시점에 이 데이터를 사용하여 R_base, T_base를 생성합니다.
CAM_DATA = np.array([
    [304.6, -156.4, 984.6, -137.2, -16.3, -110.5],
    [-219.4, -248.5, 1310.1, -164.6, -4.5, -69.9],
    [372.4, -352.2, 1022.9, -141.0, -28.8, -148.3],
    [158.1, -247.5, 1015.5, -174.2, -21.6, -10.7],
    [-70.5, -455.2, 1600.2, -159.3, -49.1, -5.7],
    [-121.4, 173.4, 1984.4, -153.0, 23.0, 8.0],
    [404.8, 245.4, 1067.0, -169.0, -53.9, 2.6],
    [421.9, -304.0, 940.4, 132.0, -1.8, 20.3],
    [-101.9, -111.4, 423.7, 121.6, 5.0, 10.9]
])

ROBOT_DATA = np.array([
    [971.42, -117.44, 969.11, 99.69, -47.15, 151.07],
    [676.39, 407.80, 848.89, 61.61, -80.63, 138.38],
    [1026.19, -4.95, 1159.95, 47.45, -22.50, -144.85],
    [830.43, -11.81, 981.49, 74.05, -68.44, -178.65],
    [1056.06, 553.84, 899.97, 96.65, -90.84, 166.60],
    [1039.0, 638.0, 245.0, 76.41, -86.71, 160.39],
    [1012.93, -284.34, 500.93, 101.28, -86.71, 160.39],
    [1012.93, -134.93, 1153.05, 85.05, -24.30, 134.89],
    [319.43, -198.54, 1153.05, 77.15, -9.18, 143.07]
])

class ArUcoRobotMaster:
    def __init__(self, root):
        self.root = root
        self.root.title("Doosan ArUco Tracker v2.0")
        self.root.geometry("500x850")
        
        self.conn = None
        self.is_tracking = False 

        # --- 캘리브레이션 연산 ---
        self.R_base, self.T_base = self.calibrate_base(CAM_DATA, ROBOT_DATA)
        
        # 현재 타겟 좌표
        self.target_pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        # --- RealSense & ArUco 초기화 ---
        self.setup_camera()
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters()

        # --- UI 구성 ---
        self.setup_ui()

        # --- 스레드 관리 ---
        self.stop_event = threading.Event()
        self.cam_thread = threading.Thread(target=self.camera_worker, daemon=True)
        self.cam_thread.start()

        self.start_server()

    def calibrate_base(self, cam_pts, robot_pts):
        A = cam_pts[:, :3]
        B = robot_pts[:, :3]
        centroid_A = np.mean(A, axis=0)
        centroid_B = np.mean(B, axis=0)
        H = np.dot((A - centroid_A).T, (B - centroid_B))
        U, S, Vt = np.linalg.svd(H)
        R_mat = np.dot(Vt.T, U.T)
        if np.linalg.det(R_mat) < 0:
            Vt[2,:] *= -1
            R_mat = np.dot(Vt.T, U.T)
        T_vec = centroid_B - np.dot(R_mat, centroid_A)
        return R_mat, T_vec

    def setup_camera(self):
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        profile = self.pipeline.start(config)
        intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.camera_matrix = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]])
        self.dist_coeffs = np.array(intr.coeffs)

    def setup_ui(self):
        tk.Label(self.root, text="[ Doosan ArUco Control ]", font=("Arial", 14, "bold")).pack(pady=15)
        
        frame_entries = tk.Frame(self.root)
        frame_entries.pack()
        self.entries = []
        labels = ['X', 'Y', 'Z', 'Rx', 'Ry', 'Rz']
        for i, txt in enumerate(labels):
            tk.Label(frame_entries, text=txt, font=("Arial", 10)).grid(row=i//3, column=(i%3)*2, pady=5)
            e = tk.Entry(frame_entries, width=10, justify='center', font=("Arial", 10))
            e.grid(row=i//3, column=(i%3)*2 + 1, padx=5)
            e.insert(0, "0.0")
            self.entries.append(e)

        self.track_btn = tk.Button(self.root, text="START AUTO TRACKING", bg="red", fg="white", 
                                   font=("Arial", 12, "bold"), height=2, command=self.toggle_tracking)
        self.track_btn.pack(fill="x", padx=50, pady=20)

        tk.Button(self.root, text="수동 좌표 이동 (MOVE_ABS)", bg="skyblue", command=self.move_abs).pack(fill="x", padx=50, pady=5)
        
        self.status = tk.Label(self.root, text="로봇 연결 대기 중...", bd=1, relief="sunken", anchor="w")
        self.status.pack(side="bottom", fill="x")

    def get_robot_pose_math(self, tvec, rvec):
        """오늘 완성한 캘리브레이션 공식 적용"""
        # 1. 위치 변환
        pos = np.dot(self.R_base, tvec) + self.T_base
        
        # [정밀 오프셋 반영]
        final_x = pos[0] - 22.57
        final_y = pos[1] + 41.75
        final_z = pos[2] + 0.45

        # 2. 자세 변환
        mat_marker = R.from_rotvec(rvec).as_matrix()
        mat_robot = np.dot(self.R_base, mat_marker)
        r_final = R.from_matrix(mat_robot)
        euler = r_final.as_euler('xyz', degrees=True)

        # [각도 오프셋 반영]
        res_rx = euler[0] - 97.89
        res_ry = euler[1] + 33.46
        res_rz = euler[2] + 175.24
        
        final_rot = [((a + 180) % 360) - 180 for a in [res_rx, res_ry, res_rz]]
        return [final_x, final_y, final_z], final_rot

    def camera_worker(self):
        while not self.stop_event.is_set():
            frames = self.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame: continue

            img = np.asanyarray(color_frame.get_data())
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)

            if ids is not None:
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 100, self.camera_matrix, self.dist_coeffs)
                
                marker_map = {ids[i][0]: i for i in range(len(ids))}
                
                # 따라갈 대상 마커 (ID: 1)
                if 1 in marker_map:
                    idx = marker_map[1]
                    # 캘리브레이션 공식으로 위치만 계산
                    pos, _ = self.get_robot_pose_math(tvecs[idx][0], rvecs[idx][0])
                    
                    # --- [각도 고정 설정] ---
                    # 로봇 펜던트에서 확인한 가장 안정적인 각도를 여기에 넣으세요.
                    # 예: 바닥을 정면으로 바라보는 각도 (A:0, B:180, C:0 등)
                    fixed_rx = 90.04
                    fixed_ry = 180.0
                    fixed_rz = -146.0
                    
                    self.target_pose = [pos[0], pos[1], pos[2], fixed_rx, fixed_ry, fixed_rz]
                    
                    # 시각화
                    cv2.aruco.drawDetectedMarkers(img, corners, ids)
                    cv2.putText(img, f"FOLLOWING ID:1 (ROT FIXED)", (20, 40), 1, 1.5, (255, 255, 0), 2)
                    cv2.putText(img, f"X:{int(pos[0])} Y:{int(pos[1])} Z:{int(pos[2])}", (20, 70), 1, 1.2, (0, 255, 0), 2)

                    if self.is_tracking and self.conn:
                        try:
                            # 고정된 각도와 함께 명령 전송
                            msg = f"MOVE,{pos[0]:.1f},{pos[1]:.1f},{pos[2]:.1f},{fixed_rx},{fixed_ry},{fixed_rz}\n"
                            self.conn.sendall(msg.encode())
                        except: pass

            cv2.imshow("ArUco Fixed-Rot View", img)
            if cv2.waitKey(1) == ord('q'): break

    def toggle_tracking(self):
        self.is_tracking = not self.is_tracking
        self.track_btn.config(text="STOP AUTO TRACKING" if self.is_tracking else "START AUTO TRACKING", 
                              bg="black" if self.is_tracking else "red")

    def start_server(self):
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind(("0.0.0.0", 30002))
        self.server.listen(1)
        self.root.after(100, self.accept_conn)

    def accept_conn(self):
        self.server.setblocking(False)
        try:
            self.conn, addr = self.server.accept()
            self.status.config(text=f"로봇 접속됨: {addr}", fg="green")
        except: self.root.after(500, self.accept_conn)

    def move_abs(self):
        if not self.conn: return
        vals = [e.get() for e in self.entries]
        self.conn.sendall(f"MOVE,{','.join(vals)}\n".encode())

if __name__ == "__main__":
    root = tk.Tk()
    app = ArUcoRobotMaster(root)
    root.mainloop()