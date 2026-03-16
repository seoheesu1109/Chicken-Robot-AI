import cv2
import cv2.aruco as aruco
import numpy as np
import time
import socket
import threading

# 1. 렌즈 보정 데이터 로드
try:
    calib_data = np.load('calibration_data.npz')
    mtx, dist = calib_data['mtx'], calib_data['dist']
    print("✅ 렌즈 보정 데이터 로드 성공")
except:
    print("❌ calibration_data.npz 파일이 없습니다. 먼저 렌즈 보정을 진행하세요.")
    exit()

# 2. 설정 값
CALIB_POINTS = [[600, 750], [850, 750], [600, 950], [850, 950], [720, 850]]
Z_HEIGHT = 1018.20 
BASE_ORI = "78.37,85.31,92.49"
ARUCO_DICT = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
ARUCO_PARAMS = aruco.DetectorParameters()

class AutoCalibrator:
    def __init__(self):
        # 1번 카메라 강제 지정 및 설정
        self.cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        self.conn = None
        self.is_connected = False
        self.start_socket_server()

    def start_socket_server(self):
        def server_thread():
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("0.0.0.0", 30002))
            s.listen(1)
            print("🚀 로봇 접속 대기 중 (포트 30002)...")
            conn, addr = s.accept()
            self.conn = conn
            self.is_connected = True
            print(f"✅ 로봇 연결됨: {addr}")
        
        threading.Thread(target=server_thread, daemon=True).start()

    def get_undistorted_frame(self):
        ret, frame = self.cap.read()
        if not ret: return None
        
        # --- 이 아래 줄을 추가해서 '보정 전' 원본을 강제로 봅니다 ---
        return frame 
        # --------------------------------------------------

        # 아래 코드는 일단 무시됩니다.
        h, w = frame.shape[:2]
        newcameramtx, _ = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
        dst = cv2.undistort(frame, mtx, dist, None, newcameramtx)
        return dst

    def run(self):
        print("💡 화면 창을 클릭한 후 'S' 키를 누르면 캘리브레이션을 시작합니다.")
        
        while True:
            frame = self.get_undistorted_frame()
            if frame is None: continue

            display_msg = "WAITING FOR ROBOT..." if not self.is_connected else "ROBOT READY! PRESS 'S' TO START"
            color = (0, 0, 255) if not self.is_connected else (0, 255, 0)
            
            # 마커 실시간 인식 테스트 (인식되는지 미리 확인용)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = aruco.detectMarkers(gray, ARUCO_DICT, parameters=ARUCO_PARAMS)
            if ids is not None:
                aruco.drawDetectedMarkers(frame, corners, ids)
                cv2.putText(frame, "MARKER DETECTED!", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

            cv2.putText(frame, display_msg, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.imshow("Calibration Monitor", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s') and self.is_connected:
                self.execute_sequence()
                break
            elif key == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def execute_sequence(self):
        print("🏁 자동 시퀀스 시작!")
        cam_pts, robot_pts = [], []

        for p in CALIB_POINTS:
            cmd = f"MOVE,{p[0]:.2f},{p[1]:.2f},{Z_HEIGHT:.2f},{BASE_ORI}"
            print(f"📍 이동 명령 전송: {p}")
            
            # 1. 로봇에게 명령을 보냄
            self.conn.sendall((cmd + "\n").encode())
            self.conn.settimeout(15.0)

            # 2. [중요] 로봇이 이동하는 동안 화면을 실시간으로 계속 갱신함
            print("   ㄴ 로봇 이동 중... 화면 갱신 대기")
            while True:
                # 카메라 버퍼 비우기 및 최신 프레임 가져오기
                for _ in range(5): self.cap.read() 
                frame = self.get_undistorted_frame()
                
                if frame is not None:
                    cv2.putText(frame, f"MOVING TO: {p}", (30, 40), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    cv2.imshow("Calibration Monitor", frame)
                
                # 로봇으로부터 'DONE' 응답이 왔는지 비차단(Non-blocking)으로 확인
                # 여기서는 간단하게 로봇 응답 대기 시간을 짧게 주어 루프를 돌립니다.
                try:
                    self.conn.setblocking(False) # 잠시 비차단 모드로 변경
                    res = self.conn.recv(1024).decode()
                    if "DONE" in res:
                        print("   ㄴ 로봇 도착 확인!")
                        break
                except BlockingIOError:
                    # 아직 데이터가 안 왔으면 화면 갱신 계속
                    pass
                
                if cv2.waitKey(1) & 0xFF == ord('q'): return # 중단 처리

            # 3. 로봇 정지 후 안정화 대기
            time.sleep(1.5) 
            self.conn.setblocking(True) # 다시 차단 모드로 복구

            # 4. 마커 좌표 추출
            frame = self.get_undistorted_frame()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = aruco.detectMarkers(gray, ARUCO_DICT, parameters=ARUCO_PARAMS)
            
            if ids is not None:
                c = corners[0][0]
                cx, cy = np.mean(c[:, 0]), np.mean(c[:, 1])
                # 원본 시야 확보를 위해 'return frame'을 쓰셨다면 아래 계산식 점검 필요
                raw_x = (cx - 640) * (500 / 900)
                raw_y = (cy - 360) * (500 / 900)
                cam_pts.append([raw_x, raw_y])
                robot_pts.append(p)
                print(f"   ㄴ ✅ 매칭 성공")
            else:
                print(f"   ㄴ ❌ 마커 인식 실패")

        # 5. 최종 행렬 저장 (이전과 동일)
        if len(cam_pts) >= 4:
            H, _ = cv2.findHomography(np.array(cam_pts), np.array(robot_pts))
            np.save("homography_matrix.npy", H)
            print("\n🎉 완료! homography_matrix.npy 파일이 생성되었습니다.")

        if len(cam_pts) >= 4:
            H, _ = cv2.findHomography(np.array(cam_pts), np.array(robot_pts))
            np.save("homography_matrix.npy", H)
            print("\n🎉 완료! homography_matrix.npy 파일이 생성되었습니다.")
        else:
            print("\n❌ 실패: 인식된 포인트가 너무 적습니다.")

    def send_and_wait(self, cmd):
        if not self.conn: return False
        try:
            self.conn.sendall((cmd + "\n").encode())
            self.conn.settimeout(15.0)
            res = self.conn.recv(1024).decode()
            return "DONE" in res
        except: return False

if __name__ == "__main__":
    AutoCalibrator().run()