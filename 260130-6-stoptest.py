import socket
import tkinter as tk
import cv2
import numpy as np
import pyrealsense2 as rs
import threading
import collections
import time
from ultralytics import YOLO, YOLOWorld

# [1] 새로운 캘리브레이션 데이터 (6점 수집 데이터)
# CAM_DATA: [u, v, depth(mm)]
CAM_DATA = np.array([
    [320, 180, 1200], [640, 180, 1350],
    [320, 360, 2400], [640, 360, 990],
    [320, 540, 2400], [640, 540, 2700]
])
# ROBOT_DATA: [x, y, z]
ROBOT_DATA = np.array([
    [896.55, 208.56, 351.00], [1333.38, 187.55, 346.26],
    [974.63, -16.52, 658.78], [1311.10, -30.07, 658.78],
    [1018.56, -156.68, 841.27], [1300.96, -174.90, 841.27]
])

# 기본 설정값
SAFE_Z_OVER = 450.0  # 이동 시 안전 높이
DROP_POS = [1460.0, -335.0, 122.0] # 배출 위치

class ChickenRobotMasterCeiling:
    def __init__(self, root):
        self.root = root
        self.conn = None
        self.is_tracking = False 
        self.is_waiting = False
        self.is_emergency = False

        # 모델 설정
        self.model = YOLO(r"C:\runs\detect\train12\weights\best.pt")
        self.safety_model = YOLOWorld('yolov8s-worldv2.pt') 
        self.safety_model.set_classes(["person"])

        self.pos_history = collections.deque(maxlen=5) 
        self.last_sent_pos = np.array([0.0, 0.0, 0.0]) 

        # 캘리브레이션 행렬 계산
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
                # 제한 영역 해제 (원하는 좌표로 그대로 전송)
                msg = f"MOVE,{x:.1f},{y:.1f},{z:.1f},0.0,180.0,0.0\n"
                self.conn.sendall(msg.encode())
                time.sleep(0.05)
            except: pass

    def send_robot_stop(self):
        if self.conn:
            try:
                self.conn.sendall("STOP\n".encode())
                print("🚨 Emergency STOP")
            except: pass

    def camera_worker(self):
        while not self.stop_event.is_set():
            frames = self.pipeline.wait_for_frames()
            aligned = self.align.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if not color_frame or not depth_frame: continue

            # 이미지 가져오기 및 상하좌우 반전
            img = np.asanyarray(color_frame.get_data())
            img = np.flip(img, axis=(0, 1)).copy()

            # [1] 안전 검사 (YOLO-World)
            safety_results = self.safety_model.predict(img, conf=0.4, verbose=False)
            person_in_zone = False
            for r in safety_results:
                for box in r.boxes:
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    cv2.rectangle(img, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 0, 255), 3)
                    person_in_zone = True # 사람 감지 시 즉시 정지 모드

            if person_in_zone:
                if not self.is_emergency:
                    self.is_emergency = True
                    self.is_tracking = False
                    self.send_robot_stop()
                    self.update_ui_emergency(True)
            else:
                if self.is_emergency:
                    self.is_emergency = False
                    self.update_ui_emergency(False)

            # [2] 치킨 검출 (YOLOv8)
            if not self.is_emergency:
                results = self.model.predict(img, conf=0.3, imgsz=1024, verbose=False)
                for r in results:
                    for box in r.boxes:
                        xyxy = box.xyxy[0].cpu().numpy().astype(int)
                        u, v = int((xyxy[0]+xyxy[2])/2), int((xyxy[1]+xyxy[3])/2)
                        
                        # 반전된 좌표를 실제 센서 좌표로 역변환하여 Depth 추출
                        real_u, real_v = 1280 - u, 720 - v
                        depth = depth_frame.get_distance(real_u, real_v)
                        
                        if 0.1 < depth < 4.0:
                            # 2D 픽셀 + Depth를 로봇 3D 좌표로 변환
                            pt_cam = np.array(rs.rs2_deproject_pixel_to_point(self.intrinsics, [real_u, real_v], depth)) * 1000.0
                            pos = np.dot(self.R_base, pt_cam) + self.T_base
                            
                            self.pos_history.append(pos)
                            smooth_pos = np.mean(self.pos_history, axis=0)

                            # 트래킹 로직
                            if self.is_tracking and not self.is_waiting:
                                dist = np.linalg.norm(smooth_pos[:2] - self.last_sent_pos[:2])
                                if dist < 8.0: # 정지 상태로 판단
                                    self.is_waiting = True
                                    threading.Timer(1.5, self.auto_drop_sequence, args=[smooth_pos]).start()
                                    break
                                elif dist > 20.0:
                                    self.send_robot_command(smooth_pos[0], smooth_pos[1], smooth_pos[2])
                                    self.last_sent_pos = smooth_pos.copy()

                            # 화면 표시
                            cv2.rectangle(img, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)
                            cv2.putText(img, f"X:{pos[0]:.0f} Y:{pos[1]:.0f}", (xyxy[0], xyxy[1]-10), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            cv2.imshow("Chicken Master Ceiling (No Limit)", img)
            if cv2.waitKey(1) == ord('q'): break

    def auto_drop_sequence(self, current_pos):
        if self.is_emergency: return
        print(f"🎯 Target Acquired: {current_pos}")
        # 1. 물체 위로 이동
        self.send_robot_command(current_pos[0], current_pos[1], SAFE_Z_OVER)
        time.sleep(1.0)
        # 2. 하강 및 집기 (이 부분에 그리퍼 동작 추가 가능)
        self.send_robot_command(current_pos[0], current_pos[1], current_pos[2])
        time.sleep(1.0)
        # 3. 안전 높이로 상승
        self.send_robot_command(current_pos[0], current_pos[1], SAFE_Z_OVER)
        time.sleep(1.0)
        # 4. 배출 지점으로 이동
        self.send_robot_command(DROP_POS[0], DROP_POS[1], SAFE_Z_OVER)
        time.sleep(2.0)
        self.send_robot_command(DROP_POS[0], DROP_POS[1], DROP_POS[2])
        time.sleep(1.0)
        # 5. 복귀
        self.send_robot_command(DROP_POS[0], DROP_POS[1], SAFE_Z_OVER)
        self.is_waiting = False

    def setup_ui(self):
        tk.Label(self.root, text="[ Chicken Robot Master v2.3 (Ceiling) ]", font=("Arial", 14, "bold")).pack(pady=20)
        self.track_btn = tk.Button(self.root, text="START TRACKING", bg="red", fg="white", 
                                   font=("Arial", 12, "bold"), height=2, command=self.toggle_tracking)
        self.track_btn.pack(fill="x", padx=50, pady=20)
        self.status = tk.Label(self.root, text="시스템 준비 완료", bd=1, relief="sunken", anchor="w")
        self.status.pack(side="bottom", fill="x")

    def toggle_tracking(self):
        if self.is_emergency: return
        self.is_tracking = not self.is_tracking
        self.track_btn.config(text="STOP" if self.is_tracking else "START TRACKING", 
                              bg="black" if self.is_tracking else "red")

    def update_ui_emergency(self, is_danger):
        if is_danger:
            self.track_btn.config(text="🚨 EMERGENCY 🚨", bg="orange")
            self.status.config(text="사람 감지: 모든 동작 중지", fg="red")
        else:
            self.track_btn.config(text="START TRACKING", bg="red")
            self.status.config(text="안전: 작업 가능", fg="blue")

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
    app = ChickenRobotMasterCeiling(root)
    root.mainloop()