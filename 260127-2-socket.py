import socket
import tkinter as tk
from tkinter import messagebox

HOST = "0.0.0.0"
PORT = 30002

class RobotMaster:
    def __init__(self, root):
        self.root = root
        self.root.title("Doosan Robot Advanced Controller")
        self.root.geometry("500x750")
        
        self.conn = None
        self.slots = [None] * 4 # 좌표 저장 슬롯 4개

        # --- 좌표 입력/표시 영역 ---
        tk.Label(root, text="[ Robot Pose ]", font=("Arial", 12, "bold")).pack(pady=10)
        self.entries = []
        frame_entries = tk.Frame(root)
        frame_entries.pack()
        
        labels = ['X', 'Y', 'Z', 'Rx', 'Ry', 'Rz']
        for i, txt in enumerate(labels):
            tk.Label(frame_entries, text=txt, width=3).grid(row=i//3, column=(i%3)*2)
            e = tk.Entry(frame_entries, width=10, justify='center')
            e.grid(row=i//3, column=(i%3)*2 + 1, padx=5, pady=5)
            e.insert(0, "0.0")
            self.entries.append(e)

        # --- 제어 버튼 ---
        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=10)
        tk.Button(btn_frame, text="현재 좌표 읽기 (GET)", bg="orange", width=20, command=self.get_pos).grid(row=0, column=0, padx=5)
        tk.Button(btn_frame, text="절대 좌표 이동 (MOVE)", bg="skyblue", width=20, command=self.move_abs).grid(row=0, column=1, padx=5)

        # --- 조그 버튼 영역 ---
        tk.Label(root, text="[ Jog Control (Step: 10mm/deg) ]", font=("Arial", 10, "bold")).pack(pady=10)
        jog_frame = tk.Frame(root)
        jog_frame.pack()
        
        jog_dirs = [
            ("X+", [10,0,0,0,0,0]), ("X-", [-10,0,0,0,0,0]),
            ("Y+", [0,10,0,0,0,0]), ("Y-", [0,-10,0,0,0,0]),
            ("Z+", [0,0,10,0,0,0]), ("Z-", [0,0,-10,0,0,0])
        ]
        for i, (name, val) in enumerate(jog_dirs):
            tk.Button(jog_frame, text=name, width=8, height=2, command=lambda v=val: self.send_jog(v)).grid(row=i//2, column=i%2, padx=5, pady=2)

        # --- 좌표 저장 슬롯 (4개) ---
        tk.Label(root, text="[ Memory Slots ]", font=("Arial", 10, "bold")).pack(pady=10)
        slot_frame = tk.Frame(root)
        slot_frame.pack()
        for i in range(4):
            tk.Label(slot_frame, text=f"Slot {i+1}:").grid(row=i, column=0, padx=5)
            tk.Button(slot_frame, text="SAVE", bg="#e1e1e1", command=lambda idx=i: self.save_slot(idx)).grid(row=i, column=1, padx=2, pady=2)
            tk.Button(slot_frame, text="GOTO", bg="#c1c1c1", command=lambda idx=i: self.goto_slot(idx)).grid(row=i, column=2, padx=2, pady=2)

        self.status = tk.Label(root, text="로봇 연결 대기 중...", bd=1, relief="sunken", anchor="w")
        self.status.pack(side="bottom", fill="x")

        self.start_server()

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
            self.status.config(text=f"연결됨: {addr}", fg="green")
        except: self.root.after(500, self.accept_conn)

    def get_pos(self):
        if not self.conn: return
        self.conn.sendall(b"GET\n")
        raw = self.conn.recv(1024).decode().strip()
        parts = raw.split(',')
        if len(parts) == 6:
            for i in range(6):
                self.entries[i].delete(0, tk.END)
                self.entries[i].insert(0, parts[i])

    def move_abs(self):
        if not self.conn: return
        vals = [e.get() for e in self.entries]
        self.conn.sendall(f"MOVE,{','.join(vals)}\n".encode())

    def send_jog(self, delta):
        if not self.conn: return
        delta_str = [str(d) for d in delta]
        self.conn.sendall(f"JOG,{','.join(delta_str)}\n".encode())

    def save_slot(self, idx):
        self.slots[idx] = [e.get() for e in self.entries]
        self.status.config(text=f"Slot {idx+1}에 좌표가 저장되었습니다.")

    def goto_slot(self, idx):
        if self.slots[idx]:
            for i in range(6):
                self.entries[i].delete(0, tk.END)
                self.entries[i].insert(0, self.slots[idx][i])
            self.move_abs()

if __name__ == "__main__":
    root = tk.Tk()
    app = RobotMaster(root)
    root.mainloop()