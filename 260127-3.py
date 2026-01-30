import socket
import tkinter as tk
import cv2
import numpy as np
import pyrealsense2 as rs
import threading

# 서버 설정 (로봇이 이 IP로 접속해야 함)
HOST = "0.0.0.0"
PORT = 30002

class RobotMaster:
    def __init__(self, root):
        self.root = root
        self.root.title("Doosan Robot Advanced Controller")
        self.root.geometry("500x800")
        
        self.conn = None
        self.is_tracking = False 

        # --- 캘리브레이션 및 좌표 변수 ---
        self.offset_x, self.offset_y, self.offset_z = 0.0, 0.0, 0.0
        self.curr_cam_x, self.curr_cam_y, self.curr_cam_z = 0.0, 0.0, 0.0
        self.target_rx, self.target_ry, self.target_rz = 0.0, 0.0, 0.0

        # --- RealSense 초기화 ---
        self.setup_camera()

        # --- UI 구성 ---
        self.setup_ui()

        # --- 스레드 관리 (3.13 안정성 확보) ---
        self.stop_event = threading.Event()
        self.cam_thread = threading.Thread(target=self.camera_worker, daemon=True)
        self.cam_thread.start()

        # 로봇 서버 시작
        self.start_server()

    def setup_camera(self):
        try:
            self.pipeline = rs.pipeline()
            self.config = rs.config()
            self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            self.profile = self.pipeline.start(self.config)
            self.align = rs.align(rs.stream.color)
            self.intrinsics = self.profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        except Exception as e:
            print(f"카메라 연결 에러: {e}")

    def setup_ui(self):
        tk.Label(self.root, text="[ Robot Pose Control ]", font=("Arial", 12, "bold")).pack(pady=10)
        self.entries = []
        frame_entries = tk.Frame(self.root)
        frame_entries.pack()
        
        labels = ['X', 'Y', 'Z', 'Rx', 'Ry', 'Rz']
        for i, txt in enumerate(labels):
            tk.Label(frame_entries, text=txt, width=3).grid(row=i//3, column=(i%3)*2)
            e = tk.Entry(frame_entries, width=12, justify='center')
            e.grid(row=i//3, column=(i%3)*2 + 1, padx=5, pady=5)
            e.insert(0, "0.0")
            self.entries.append(e)

        # 핵심 제어 버튼
        tk.Button(self.root, text="현재 좌표 읽기 & 캘리브레이션", bg="orange", font=("Arial", 10, "bold"),
                  height=2, command=self.get_pos_and_cal).pack(fill="x", padx=50, pady=10)

        self.track_btn = tk.Button(self.root, text="START AUTO TRACKING", bg="red", fg="white", 
                                   font=("Arial", 10, "bold"), height=2, command=self.toggle_tracking)
        self.track_btn.pack(fill="x", padx=50, pady=5)

        tk.Button(self.root, text="수동 좌표로 이동 (MOVE)", bg="skyblue", command=self.move_abs).pack(fill="x", padx=50, pady=5)

        # 상태 표시바
        self.status = tk.Label(self.root, text="로봇 연결 대기 중...", bd=1, relief="sunken", anchor="w")
        self.status.pack(side="bottom", fill="x")

    def camera_worker(self):
        """별도 스레드에서 카메라 연산 및 OpenCV 출력 (GIL 에러 방지)"""
        try:
            while not self.stop_event.is_set():
                frames = self.pipeline.wait_for_frames()
                aligned_frames = self.align.process(frames)
                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()

                if not color_frame or not depth_frame: continue

                img = np.asanyarray(color_frame.get_data())
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                
                # 빨간색 검출 범위 (HSV)
                mask = cv2.inRange(hsv, np.array([0, 120, 70]), np.array([10, 255, 255])) + \
                       cv2.inRange(hsv, np.array([170, 120, 70]), np.array([180, 255, 255]))
                
                # 정보창 배경
                cv2.rectangle(img, (10, 10), (320, 110), (0, 0, 0), -1)
                state_color = (0, 0, 255) if not self.is_tracking else (0, 255, 0)
                cv2.putText(img, f"AUTO TRACK: {'ON' if self.is_tracking else 'OFF'}", (20, 35), 1, 1.2, state_color, 2)

                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    c = max(contours, key=cv2.contourArea)
                    if cv2.contourArea(c) > 1000:
                        M = cv2.moments(c)
                        if M["m00"] != 0:
                            cx, cy = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
                            dist = depth_frame.get_distance(cx, cy)
                            if dist > 0:
                                pt = rs.rs2_deproject_pixel_to_point(self.intrinsics, [cx, cy], dist)
                                self.curr_cam_x, self.curr_cam_y, self.curr_cam_z = [p * 1000 for p in pt]
                                
                                # 최종 명령 좌표 계산
                                self.target_rx = self.curr_cam_z + self.offset_x
                                self.target_ry = -self.curr_cam_x + self.offset_y
                                self.target_rz = -self.curr_cam_y + self.offset_z
                                
                                # 실시간 화면 텍스트
                                cv2.putText(img, f"GO -> X:{int(self.target_rx)} Y:{int(self.target_ry)} Z:{int(self.target_rz)}", 
                                            (20, 65), 1, 1.0, (255, 255, 0), 2)
                                cv2.putText(img, f"OFF -> X:{int(self.offset_x)} Y:{int(self.offset_y)}", 
                                            (20, 95), 1, 0.9, (150, 150, 150), 1)
                                
                                if self.is_tracking and self.conn:
                                    try:
                                        msg = f"MOVE,{self.target_rx:.1f},{self.target_ry:.1f},{self.target_rz:.1f},0,180,0\n"
                                        self.conn.sendall(msg.encode())
                                    except: pass

                                cv2.circle(img, (cx, cy), 10, (0, 0, 255), -1)

                cv2.imshow("RealSense Robot View", img)
                if cv2.waitKey(1) & 0xFF == ord('q'): break
        finally:
            self.pipeline.stop()
            cv2.destroyAllWindows()

    def get_pos_and_cal(self):
        """[중요] 버튼 클릭 시 로봇 좌표를 읽어와 카메라와 동기화"""
        if not self.conn: 
            messagebox.showwarning("연결 에러", "로봇이 연결되지 않았습니다.")
            return
        try:
            self.conn.sendall(b"GET\n")
            raw = self.conn.recv(1024).decode().strip()
            p = [float(v) for v in raw.split(',')]
            if len(p) >= 3:
                for i in range(len(p)):
                    self.entries[i].delete(0, tk.END)
                    self.entries[i].insert(0, str(p[i]))
                
                # 영점(Offset) 계산: 로봇 현재 위치 - 카메라가 보고 있는 물체 위치
                self.offset_x = p[0] - self.curr_cam_z
                self.offset_y = p[1] - (-self.curr_cam_x)
                self.offset_z = p[2] - (-self.curr_cam_y)
                
                self.status.config(text=f"캘리브레이션 완료! Offset 적용됨", fg="blue")
        except Exception as e:
            print(f"GET Error: {e}")

    def toggle_tracking(self):
        self.is_tracking = not self.is_tracking
        self.track_btn.config(text="STOP AUTO TRACKING" if self.is_tracking else "START AUTO TRACKING", 
                              bg="black" if self.is_tracking else "red")

    def start_server(self):
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind((HOST, PORT))
        self.server.listen(1)
        self.root.after(100, self.accept_conn)

    def accept_conn(self):
        self.server.setblocking(False)
        try:
            self.conn, addr = self.server.accept()
            self.status.config(text=f"로봇 연결됨: {addr}", fg="green")
        except: self.root.after(500, self.accept_conn)

    def move_abs(self):
        if not self.conn: return
        vals = [e.get() for e in self.entries]
        self.conn.sendall(f"MOVE,{','.join(vals)}\n".encode())

if __name__ == "__main__":
    from tkinter import messagebox
    root = tk.Tk()
    app = RobotMaster(root)
    root.mainloop()