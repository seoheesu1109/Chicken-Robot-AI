import socket
import tkinter as tk
from tkinter import messagebox

HOST = "0.0.0.0"
PORT = 30002

class DoosanApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Doosan Robot Controller")
        self.root.geometry("400x500")
        
        self.conn = None
        self.buffer = ""

        # --- UI 구성 ---
        labels = ['X', 'Y', 'Z', 'Rx', 'Ry', 'Rz']
        self.entries = []
        
        for i, txt in enumerate(labels):
            tk.Label(root, text=f"{txt}:", font=("Arial", 10)).grid(row=i, column=0, padx=20, pady=10)
            e = tk.Entry(root, width=20, justify='center')
            e.grid(row=i, column=1)
            e.insert(0, "0.0")
            self.entries.append(e)

        # 버튼들
        tk.Button(root, text="현재 좌표 읽기 (GET)", bg="#FF9800", height=2, command=self.get_pos).grid(row=6, column=0, columnspan=2, sticky="ew", padx=30, pady=10)
        tk.Button(root, text="로봇으로 전송 (MOVE)", bg="#2196F3", fg="white", height=2, command=self.move_robot).grid(row=7, column=0, columnspan=2, sticky="ew", padx=30, pady=10)

        # 상태바
        self.status = tk.Label(root, text="연결 대기 중...", bd=1, relief="sunken", anchor="w")
        self.status.grid(row=8, column=0, columnspan=2, sticky="we", pady=20)

        self.start_server()

    def start_server(self):
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind((HOST, PORT))
        self.server.listen(1)
        self.root.after(100, self.check_connection)

    def check_connection(self):
        self.server.setblocking(False)
        try:
            self.conn, addr = self.server.accept()
            self.status.config(text=f"연결됨: {addr}", fg="green")
        except:
            self.root.after(500, self.check_connection)

    def get_pos(self):
        if not self.conn: return
        try:
            self.conn.sendall(b"GET\n")
            raw_data = self.conn.recv(1024).decode()
            
            # 줄바꿈 기준으로 데이터 분리 및 검증
            lines = raw_data.strip().split('\n')
            if not lines: return
            
            parts = lines[-1].split(',') # 가장 최신 줄 선택
            if len(parts) == 6:
                for i in range(6):
                    self.entries[i].delete(0, tk.END)
                    self.entries[i].insert(0, parts[i])
                self.status.config(text="좌표 로드 성공", fg="blue")
            else:
                self.status.config(text="데이터 수신 오류", fg="red")
        except Exception as e:
            self.status.config(text=f"에러: {e}")

    def move_robot(self):
        if not self.conn: return
        try:
            # 입력창의 값들을 모아서 전송
            vals = [e.get() for e in self.entries]
            msg = "MOVE," + ",".join(vals) + "\n"
            self.conn.sendall(msg.encode())
            self.status.config(text="이동 명령 전송됨...", fg="black")
            
            # 로봇의 DONE 응답 대기 (선택 사항)
            # self.conn.recv(1024) 
        except Exception as e:
            messagebox.showerror("오류", f"전송 실패: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = DoosanApp(root)
    root.mainloop()