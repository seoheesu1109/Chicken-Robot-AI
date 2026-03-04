import cv2
import numpy as np
import socket
import time

# [PC 서버 설정]
HOST = '0.0.0.0'
PORT = 30002
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((HOST, PORT))
server_socket.listen(1)
server_socket.setblocking(False)

print("📡 PC 서버 시작. 로봇 연결 대기 중...")
print("💡 키보드 'G'를 누르면 강제로 로봇에 GO 신호를 보냅니다.")
client_conn = None
detection_start_time = None
DETECTION_DURATION = 5.0 
is_sent = False

cap = cv2.VideoCapture(1) 

while True:
    if client_conn is None:
        try:
            client_conn, addr = server_socket.accept()
            print(f"✅ 로봇 연결됨: {addr}")
        except BlockingIOError: pass

    ret, frame = cap.read()
    if not ret: break

    # ROI 및 색상 감지 (기존 로직 동일)
    h, w, _ = frame.shape
    roi = frame[h//4:3*h//4, w//4:3*w//4]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower_orange = np.array([5, 80, 120]) 
    upper_orange = np.array([20, 255, 255])
    mask = cv2.inRange(hsv, lower_orange, upper_orange)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    orange_found = any(cv2.contourArea(c) > 800 for c in contours)

    # [자동 감지 로직]
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

    # [수동 버튼 및 종료 제어]
    cv2.rectangle(frame, (w//4, h//4), (3*w//4, 3*h//4), (255, 0, 0), 2)
    cv2.imshow('PC SERVER', frame)
    
    key = cv2.waitKey(1) & 0xFF
    
    # 'g' 키를 누르면 즉시 전송 (수동 버튼)
    if key == ord('g'):
        if client_conn:
            try:
                client_conn.sendall("GO,\n".encode())
                print("🔘 수동 신호 전송: GO, (G key pressed)")
                is_sent = True # 중복 전송 방지 (필요 시 주석 처리)
            except:
                print("❌ 로봇 연결 확인 필요")
                client_conn = None
        else:
            print("⚠️ 로봇이 연결되어 있지 않습니다.")

    # 'q' 키를 누르면 종료
    if key == ord('q'): break

cap.release()
cv2.destroyAllWindows()