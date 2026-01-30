# ===========================
# 기본 라이브러리 및 외부 모듈
# ===========================
import socket                  # 로봇 컨트롤러와 TCP 통신 sockek통신은 통신방식,tcp/io는 소켓이 사용하는 전송 프로토콜
import tkinter as tk            # 간단한 GUI (Start / Stop)
import cv2                      # 카메라 영상 처리
import numpy as np              # 수치 연산
import pyrealsense2 as rs       # Intel RealSense SDK
import threading                # 멀티스레드 (카메라 / 서버 분리)
import collections              # 좌표 히스토리(평균 필터)
import time
from ultralytics import YOLO    # YOLOv8 객체 탐지 모델


# ============================================================
# [1] 캘리브레이션 데이터
#   - CAM_DATA   : 카메라 좌표계 기준 3D 위치
#   - ROBOT_DATA : 로봇 베이스 좌표계 기준 3D 위치
#   - 두 좌표계를 대응시켜 변환 행렬(R, T)을 계산
#   - 로봇 끝단에 아르코 마커를 붙여 카메라와 로봇의 좌표를 맞춰 캘리브레이션 진행
# ============================================================
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


# ============================================================
# [2] 로봇 안전 작업 범위 (하드 리미트)
#   - 로봇이 절대 벗어나면 안 되는 XYZ 범위
#   - 제한영역 설정
# ============================================================
X_MIN_LIMIT, X_MAX_LIMIT = 366.0, 1600.0 
Y_MIN_LIMIT, Y_MAX_LIMIT = -484.62, 900.0
Z_MIN_LIMIT, Z_MAX_LIMIT = 60.0, 1018.42


# ============================================================
# [3] 벽(장애물) 정보
#   - XY 평면에 벽이 존재
#   - Z축을 충분히 올려서(450mm) 넘어가야 함
#   - 로봇 경로의 솥을 좌표설정해 가상공간 설정
# ============================================================
WALL_X = (508.88, 1489.08)
WALL_Y = (59.99, 572.37)
WALL_Z_TOP = 319.54
SAFE_Z_OVER = 450.0


# ============================================================
# [4] 배출(Drop) 위치
#   - 로봇이 치킨 피킹 후 배출하는 위치
# ============================================================
DROP_POS = [1460.0, -335.0, 122.0]


class ChickenRobotMaster:
    """
    YOLO + RealSense 기반 치킨 자동 추적 & 배출 로봇 마스터 클래스
    """

    def __init__(self, root):
        self.root = root
        self.conn = None                 # 로봇 TCP 연결
        self.is_tracking = False         # 추적 ON/OFF
        self.is_waiting = False          # 도달 후 2초 대기 상태

        # ------------------------------
        # YOLO 모델 및 위치 필터
        #   - 치킨 사진을 직접 찍어 yolo 모델 생성함
        # ------------------------------
        self.model = YOLO(r"C:\runs\detect\train12\weights\best.pt")
        self.pos_history = collections.deque(maxlen=5)  # 좌표 이동 평균
        self.last_sent_pos = np.array([400.0, 0.0, SAFE_Z_OVER])
        self.threshold = 15.0            # 불필요한 미세 이동 방지

        # ------------------------------
        # 카메라-로봇 좌표계 캘리브레이션
        # ------------------------------
        self.R_base, self.T_base = self.calibrate_base(CAM_DATA, ROBOT_DATA)

        self.setup_camera()
        self.setup_ui()

        # ------------------------------
        # 멀티스레드 구성
        # ------------------------------
        self.stop_event = threading.Event()
        self.cam_thread = threading.Thread(
            target=self.camera_worker, daemon=True
        )
        self.cam_thread.start()

        self.start_server()


    # ============================================================
    # 카메라 좌표 → 로봇 베이스 좌표 변환 행렬 계산
    # (SVD 기반 최적 회전 + 이동)
    # ============================================================
    def calibrate_base(self, cam_pts, robot_pts):
        A, B = cam_pts[:, :3], robot_pts[:, :3]
        cA, cB = np.mean(A, axis=0), np.mean(B, axis=0)

        H = np.dot((A - cA).T, (B - cB))
        U, S, Vt = np.linalg.svd(H)

        R_mat = np.dot(Vt.T, U.T)

        # 좌표계 반전 방지
        if np.linalg.det(R_mat) < 0:
            Vt[2, :] *= -1
            R_mat = np.dot(Vt.T, U.T)

        T_vec = cB - np.dot(R_mat, cA)
        return R_mat, T_vec


    # ============================================================
    # RealSense 카메라 초기화
    # ============================================================
    def setup_camera(self):
        self.pipeline = rs.pipeline()
        config = rs.config()

        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)

        profile = self.pipeline.start(config)

        # Depth → Color 정렬
        self.align = rs.align(rs.stream.color)

        # 카메라 내부 파라미터
        self.intrinsics = (
            profile.get_stream(rs.stream.color)
            .as_video_stream_profile()
            .get_intrinsics()
        )


    # ============================================================
    # 로봇에게 이동 명령 전송 (TCP)
    # ============================================================
    def send_robot_command(self, x, y, z):
        if not self.conn:
            return

        try:
            # 안전 범위 클램핑
            x = max(min(x, X_MAX_LIMIT), X_MIN_LIMIT)
            y = max(min(y, Y_MAX_LIMIT), Y_MIN_LIMIT)
            z = max(min(z, Z_MAX_LIMIT), Z_MIN_LIMIT)

            msg = f"MOVE,{x:.1f},{y:.1f},{z:.1f},0.0,180.0,0.0\n"
            self.conn.sendall(msg.encode())
            time.sleep(0.05)

        except:
            pass


    # ============================================================
    # 현재 XY가 벽 영역에 포함되는지 판단
    # ============================================================
    def is_in_wall_plane(self, x, y):
        return (
            WALL_X[0] <= x <= WALL_X[1] and
            WALL_Y[0] <= y <= WALL_Y[1]
        )


    # ============================================================
    # 벽을 고려한 지능형 이동
    # ============================================================
    def execute_smart_move(self, target_pos):
        in_wall_now = self.is_in_wall_plane(
            self.last_sent_pos[0], self.last_sent_pos[1]
        )
        in_wall_dest = self.is_in_wall_plane(
            target_pos[0], target_pos[1]
        )

        # 벽 통과 시 → Z축 상승 → 이동 → 하강
        if in_wall_now or in_wall_dest:
            self.send_robot_command(
                self.last_sent_pos[0],
                self.last_sent_pos[1],
                SAFE_Z_OVER
            )
            self.send_robot_command(
                target_pos[0],
                target_pos[1],
                SAFE_Z_OVER
            )
            self.send_robot_command(
                target_pos[0],
                target_pos[1],
                target_pos[2]
            )
        else:
            self.send_robot_command(
                target_pos[0],
                target_pos[1],
                target_pos[2]
            )

        self.last_sent_pos = target_pos.copy()


    # ============================================================
    # 카메라 + YOLO 실시간 추적 스레드
    # ============================================================
    def camera_worker(self):
        while not self.stop_event.is_set():
            frames = self.pipeline.wait_for_frames()
            aligned = self.align.process(frames)

            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            img = np.asanyarray(color_frame.get_data())

            # YOLO 추론
            #   - conf는 신뢰도 현재 70%
            results = self.model.predict(
                img, conf=0.7, imgsz=1024, verbose=False
            )

            for r in results:
                for box in r.boxes:
                    # 바운딩 박스 중심
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    u = int((xyxy[0] + xyxy[2]) / 2)
                    v = int((xyxy[1] + xyxy[3]) / 2)

                    depth = depth_frame.get_distance(u, v)

                    # 유효 깊이 범위
                    if 0.1 < depth < 2.0:
                        # 픽셀 → 카메라 좌표(mm)
                        pt_cam = np.array(
                            rs.rs2_deproject_pixel_to_point(
                                self.intrinsics, [u, v], depth
                            )
                        ) * 1000.0

                        # 카메라 → 로봇 베이스 좌표
                        pos = np.dot(self.R_base, pt_cam) + self.T_base

                        # 실험적 보정값
                        pos[0] -= 22.57
                        pos[1] += 41.75
                        pos[2] -= 100.45

                        # 이동 평균
                        self.pos_history.append(pos)
                        smooth_pos = np.mean(self.pos_history, axis=0)

                        # 추적 중 & 대기 상태 아닐 때만 이동
                        if self.is_tracking and self.conn and not self.is_waiting:
                            dist = np.linalg.norm(
                                smooth_pos - self.last_sent_pos
                            )

                            # 도달 판정
                            if dist < 5.0:
                                self.is_waiting = True
                                threading.Timer(
                                    2.0,
                                    self.auto_drop_sequence,
                                    args=[smooth_pos]
                                ).start()
                                break

                            if dist > self.threshold:
                                self.execute_smart_move(smooth_pos)

            cv2.imshow(
                "Chicken Smart Tracker",
                cv2.resize(img, (1280, 720))
            )

            if cv2.waitKey(1) == ord('q'):
                break


    # ============================================================
    # 배출 시퀀스 (상승 → 이동 → 하강)
    # ============================================================
    def auto_drop_sequence(self, current_pos):
        # 1. 안전 높이 상승
        self.send_robot_command(
            current_pos[0], current_pos[1], SAFE_Z_OVER
        )
        time.sleep(1.2)

        # 2. 배출 위치 상공 이동
        self.send_robot_command(
            DROP_POS[0], DROP_POS[1], SAFE_Z_OVER
        )
        time.sleep(3.0)

        # 3. 하강하여 배출
        self.send_robot_command(
            DROP_POS[0], DROP_POS[1], DROP_POS[2]
        )
        time.sleep(2.0)

        # 복귀
        self.send_robot_command(
            DROP_POS[0], DROP_POS[1], SAFE_Z_OVER
        )
        time.sleep(1.2)

        self.last_sent_pos = np.array(
            [DROP_POS[0], DROP_POS[1], SAFE_Z_OVER]
        )
        self.is_waiting = False


    # ============================================================
    # UI 구성
    # ============================================================
    def setup_ui(self):
        tk.Label(
            self.root,
            text="[ YOLOv8 Chicken Smart Tracker ]",
            font=("Arial", 14, "bold")
        ).pack(pady=20)

        self.track_btn = tk.Button(
            self.root,
            text="START TRACKING",
            bg="red",
            fg="white",
            font=("Arial", 12, "bold"),
            height=2,
            command=self.toggle_tracking
        )
        self.track_btn.pack(fill="x", padx=50, pady=20)

        self.status = tk.Label(
            self.root,
            text="연결 대기 중...",
            bd=1,
            relief="sunken",
            anchor="w"
        )
        self.status.pack(side="bottom", fill="x")


    def toggle_tracking(self):
        self.is_tracking = not self.is_tracking
        self.track_btn.config(
            text="STOP" if self.is_tracking else "START TRACKING",
            bg="black" if self.is_tracking else "red"
        )


    # ============================================================
    # 로봇 TCP 서버
    # ============================================================
    def start_server(self):
        self.server = socket.socket(
            socket.AF_INET, socket.SOCK_STREAM
        )
        self.server.setsockopt(
            socket.SOL_SOCKET, socket.SO_REUSEADDR, 1
        )
        self.server.bind(("0.0.0.0", 30002))
        self.server.listen(1)
        self.root.after(100, self.accept_conn)


    def accept_conn(self):
        self.server.setblocking(False)
        try:
            self.conn, addr = self.server.accept()
            self.status.config(
                text=f"CONNECTED: {addr}",
                fg="green"
            )
        except:
            self.root.after(500, self.accept_conn)


# ============================================================
# 프로그램 시작
# ============================================================
if __name__ == "__main__":
    root = tk.Tk()
    root.title("Chicken Robot Master v2.0")
    app = ChickenRobotMaster(root)
    root.mainloop()
