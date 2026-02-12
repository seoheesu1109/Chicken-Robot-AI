import tkinter as tk
import socket
import threading
import time

class ChickenForceControl:
    def __init__(self, root):
        self.root = root
        self.root.title("Doosan Force Server")
        self.root.geometry("450x450")
        self.conn = None
        self.is_moving = False

        self.status = tk.Label(root, text="로봇 대기 중...", fg="orange", font=("Arial", 12, "bold"))
        self.status.pack(pady=10)

        # 버튼 순서대로 실행 유도
        tk.Button(root, text="1. 스캔 위치 이동 (MOVE)", command=lambda: self.run_thread(self.move_scan), height=2, width=30).pack(pady=5)
        tk.Button(root, text="2. 힘 감지 하강 (DOWN)", command=lambda: self.run_thread(self.down_force), height=2, width=30, bg="orange").pack(pady=5)
        tk.Button(root, text="3. 잡기/놓기 테스트", command=lambda: self.run_thread(self.release_gripper), height=2, width=30).pack(pady=5)
        
        self.log = tk.Text(root, height=12, width=55, bg="#f0f0f0")
        self.log.pack(pady=10)

        threading.Thread(target=self.start_server, daemon=True).start()

    def start_server(self):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.bind(("0.0.0.0", 30002))
        server.listen(1)
        while True:
            self.conn, addr = server.accept()
            self.root.after(0, lambda: self.status.config(text=f"✅ 로봇 연결됨: {addr[0]}", fg="blue"))

    def add_log(self, msg):
        self.log.insert(tk.END, f"[{time.strftime('%H:%M:%S')}] {msg}\n")
        self.log.see(tk.END)

    def run_thread(self, func):
        if self.is_moving:
            self.add_log("⚠️ 동작 중... 잠시만 기다리세요.")
            return
        threading.Thread(target=func, daemon=True).start()

    def send_and_wait(self, cmd):
        if not self.conn: 
            self.add_log("❌ 연결 없음!")
            return
        self.is_moving = True
        try:
            self.conn.sendall((cmd + "\n").encode())
            self.add_log(f"➡️ 전송: {cmd}")
            # 로봇의 DONE/ERROR 응답 대기
            data = self.conn.recv(1024) 
            if data:
                resp = data.decode().strip()
                self.add_log(f"⬅️ 응답: {resp}")
        except Exception as e:
            self.add_log(f"❌ 에러: {e}")
        finally:
            self.is_moving = False

    def move_scan(self):
        self.send_and_wait("MOVE,728.90,947.73,372.51,0.0,180.0,0.0")

    def down_force(self):
        self.send_and_wait("DOWN,728.90,947.73,50.0,0.0,180.0,0.0")

    def release_gripper(self):
        self.send_and_wait("RELEASE")

if __name__ == "__main__":
    root = tk.Tk()
    app = ChickenForceControl(root)
    root.mainloop()