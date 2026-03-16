import socket
import tkinter as tk
from tkinter import messagebox

HOST = "0.0.0.0"
PORT = 30002

class DoosanApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Doosan Robot Controller")
        self.root.geometry("400x550")
        
        self.conn = None
        self.captured_list = []  # 캡쳐된 좌표들을 저장할 리스트

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
        tk.Button(root, text="현재 좌표 읽기 & 캡쳐 (GET)", bg="#FF9800", height=2, 
                  command=self.get_pos).grid(row=6, column=0, columnspan=2, sticky="ew", padx=30, pady=10)
        
        tk.Button(root, text="로봇으로 전송 (MOVE)", bg="#2196F3", fg="white", height=2, 
                  command=self.move_robot).grid(row=7, column=0, columnspan=2, sticky="ew", padx=30, pady=10)

        # 상태바
        self.status = tk.Label(root, text="연결 대기 중...", bd=1, relief="sunken", anchor="w")
        self.status.grid(row=8, column=0, columnspan=2, sticky="we", pady=20)

        self.start_server()
        
        # 종료 시 터미널에 전체 결과 출력을 위해 설정
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

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
            
            lines = raw_data.strip().split('\n')
            if not lines: return
            
            parts = lines[-1].split(',') 
            if len(parts) == 6:
                # 1. UI 업데이트
                for i in range(6):
                    self.entries[i].delete(0, tk.END)
                    self.entries[i].insert(0, parts[i])
                
                # 2. 터미널 실시간 출력 (리스트 형식)
                float_parts = [float(p) for p in parts]
                self.captured_list.append(float_parts)
                print(f"📍 Captured: {float_parts}")
                
                self.status.config(text="좌표 로드 및 캡쳐 성공", fg="blue")
            else:
                self.status.config(text="데이터 수신 오류", fg="red")
        except Exception as e:
            self.status.config(text=f"에러: {e}")

    def move_robot(self):
        if not self.conn: return
        try:
            vals = [e.get() for e in self.entries]
            msg = "MOVE," + ",".join(vals) + "\n"
            self.conn.sendall(msg.encode())
            self.status.config(text="이동 명령 전송됨...", fg="black")
        except Exception as e:
            messagebox.showerror("오류", f"전송 실패: {e}")

    def on_closing(self):
        # 종료할 때 지금까지 모인 좌표들을 "/"로 구분해서 터미널에 뿌려줌
        if self.captured_list:
            print("\n" + "="*50)
            print("🚀 최종 캡쳐 데이터 (Gemini 전달용)")
            formatted_data = "/".join([",".join(map(str, p)) for p in self.captured_list])
            print(formatted_data)
            print("="*50 + "\n")
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = DoosanApp(root)
    root.mainloop()