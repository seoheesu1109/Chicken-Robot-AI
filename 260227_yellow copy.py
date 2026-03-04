import cv2
import mediapipe as mp
import numpy as np
import socket
import time

# [MediaPipe 초기화 - 손 감지용]
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# [PC 서버 설정]
HOST = '0.0.0.0'
PORT = 30002
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((HOST, PORT))
server_socket.listen(1)
server_socket.setblocking(False)

print("📡 통합 안전/조리 서버 시작. 로봇 연결 대기 중...")
client_conn = None
detection_start_time = None
DETECTION_DURATION = 5.0  # 주황색 감지 유지 시간
is_sent = False
is_stopped = False

cap = cv2.VideoCapture(1)

while True:
    if client_conn is None:
        try:
            client_conn, addr = server_socket.accept()
            print(f"✅ 로봇 연결됨: {addr}")
        except BlockingIOError: pass

    ret, frame = cap.read()
    if not ret: break

    # 1. MediaPipe 손 감지 (안전 최우선)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    hand_detected = results.multi_hand_landmarks is not None

    # 2. 주황색 물체 감지 (ROI 설정)
    h, w, _ = frame.shape
    roi = frame[h//4:3*h//4, w//4:3*w//4]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower_orange = np.array([5, 80, 120]) 
    upper_orange = np.array([20, 255, 255])
    mask = cv2.inRange(hsv, lower_orange, upper_orange)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    orange_found = any(cv2.contourArea(c) > 800 for c in contours)

    # --- 제어 로직 시작 ---

    # A. 손바닥이 감지된 경우 (무조건 STOP)
    if hand_detected:
        if not is_stopped: # 아직 STOP 신호를 안 보냈을 때만 전송
            if client_conn:
                try:
                    client_conn.sendall("STOP,\n".encode())
                    print("🚨 [일시정지] 손 감지됨!")
                    is_stopped = True
                except: client_conn = None
        
        cv2.putText(frame, "HOLD: HAND DETECTED", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    # [상황 B] 손바닥이 사라졌을 때 -> 재개 명령
    else:
        if is_stopped: # 이전에 정지 신호를 보냈었다면 다시 시작 신호 전송
            if client_conn:
                try:
                    client_conn.sendall("RESUME,\n".encode())
                    print("▶️ [재개] 안전 확보됨!")
                    is_stopped = False
                except: client_conn = None

        # 조리 시작(GO) 신호 감지 (안전할 때만 작동)
        if orange_found:
            if detection_start_time is None:
                detection_start_time = time.time()
        
            elapsed = time.time() - detection_start_time
            remaining = max(0, DETECTION_DURATION - elapsed)
            cv2.putText(frame, f"WATCHING: {remaining:.1f}s", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            if elapsed >= DETECTION_DURATION and not is_sent:
                if client_conn:
                    try:
                        client_conn.sendall("GO,\n".encode())
                        print("🚀 자동 신호 전송: GO,")
                        is_sent = True 
                    except: client_conn = None
        else:
            detection_start_time = None
            is_sent = False 

    # C. 수동 제어 및 종료
    cv2.rectangle(frame, (w//4, h//4), (3*w//4, 3*h//4), (255, 0, 0), 2)
    cv2.imshow('Safety & Cook Monitor', frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('g'): # 강제 GO
        if client_conn:
            client_conn.sendall("GO,\n".encode())
            print("🔘 수동 GO 전송")
            is_sent = True
    elif key == ord('q'): break

cap.release()
cv2.destroyAllWindows()