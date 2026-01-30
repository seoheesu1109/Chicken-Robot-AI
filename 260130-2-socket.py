# ============================
# 표준 라이브러리 import
# ============================

import socket
# → TCP/IP 네트워크 통신을 위한 파이썬 표준 소켓 라이브러리

import tkinter as tk
# → 파이썬 기본 GUI 라이브러리 (버튼, 입력창, 라벨 등)

from tkinter import messagebox
# → 경고창, 알림창 등을 띄울 수 있는 Tkinter 모듈
# (현재 코드에서는 사용 안 하지만 확장 대비)


# ============================
# 서버 네트워크 설정
# ============================

HOST = "0.0.0.0"
# → 모든 네트워크 인터페이스에서 접속 허용
# → 로봇 컨트롤러가 어느 IP에서 접속하든 허용

PORT = 30002
# → 로봇 컨트롤러와 통신할 TCP 포트 번호
# → 두산 로봇에서 자주 사용하는 사용자 포트


class RobotMaster:
    """
    두산 로봇을 TCP/IP로 제어하기 위한 GUI 컨트롤러 클래스
    - 현재 좌표 읽기
    - 절대 좌표 이동
    - 조그(JOG) 이동
    - 좌표 저장 / 호출
    """

    def __init__(self, root):
        # root: Tkinter 최상위 윈도우 객체
        self.root = root

        # 프로그램 창 제목 설정
        self.root.title("Doosan Robot Advanced Controller")

        # 창 크기 설정 (가로 x 세로)
        self.root.geometry("500x750")

        # ============================
        # 네트워크 관련 변수
        # ============================

        self.conn = None
        # → 실제 로봇과 연결된 TCP 소켓 객체
        # → 연결 전에는 None

        # ============================
        # 좌표 저장 슬롯
        # ============================

        self.slots = [None] * 4
        # → 좌표 저장용 슬롯 4개
        # → 각 슬롯에는 [X, Y, Z, Rx, Ry, Rz] 문자열 리스트가 들어감


        # ====================================================
        # [1] 로봇 좌표 표시 / 입력 영역
        # ====================================================

        tk.Label(
            root,
            text="[ Robot Pose ]",
            font=("Arial", 12, "bold")
        ).pack(pady=10)
        # → 상단에 "Robot Pose" 제목 표시

        self.entries = []
        # → 6개의 Entry 위젯을 담을 리스트

        frame_entries = tk.Frame(root)
        # → 좌표 입력창을 묶을 프레임 생성

        frame_entries.pack()

        labels = ['X', 'Y', 'Z', 'Rx', 'Ry', 'Rz']
        # → 로봇 포즈 구성 요소

        for i, txt in enumerate(labels):
            # 좌표 이름 라벨 (X, Y, Z, Rx, Ry, Rz)

            tk.Label(
                frame_entries,
                text=txt,
                width=3
            ).grid(
                row=i//3,
                column=(i % 3) * 2
            )
            # → 2행 3열 배치

            e = tk.Entry(
                frame_entries,
                width=10,
                justify='center'
            )
            # → 숫자 입력용 Entry 생성

            e.grid(
                row=i//3,
                column=(i % 3) * 2 + 1,
                padx=5,
                pady=5
            )

            e.insert(0, "0.0")
            # → 기본값 0.0 설정

            self.entries.append(e)
            # → Entry를 리스트에 저장 (나중에 한꺼번에 처리)


        # ====================================================
        # [2] 기본 제어 버튼 영역
        # ====================================================

        btn_frame = tk.Frame(root)
        # → 버튼들을 묶을 프레임

        btn_frame.pack(pady=10)

        tk.Button(
            btn_frame,
            text="현재 좌표 읽기 (GET)",
            bg="orange",
            width=20,
            command=self.get_pos
        ).grid(row=0, column=0, padx=5)
        # → GET 명령 전송
        # → 로봇으로부터 현재 좌표 수신

        tk.Button(
            btn_frame,
            text="절대 좌표 이동 (MOVE)",
            bg="skyblue",
            width=20,
            command=self.move_abs
        ).grid(row=0, column=1, padx=5)
        # → MOVE 명령 전송
        # → Entry에 입력된 좌표로 이동


        # ====================================================
        # [3] 조그(JOG) 제어 영역
        # ====================================================

        tk.Label(
            root,
            text="[ Jog Control (Step: 10mm/deg) ]",
            font=("Arial", 10, "bold")
        ).pack(pady=10)

        jog_frame = tk.Frame(root)
        jog_frame.pack()

        jog_dirs = [
            ("X+", [10, 0, 0, 0, 0, 0]),
            ("X-", [-10, 0, 0, 0, 0, 0]),
            ("Y+", [0, 10, 0, 0, 0, 0]),
            ("Y-", [0, -10, 0, 0, 0, 0]),
            ("Z+", [0, 0, 10, 0, 0, 0]),
            ("Z-", [0, 0, -10, 0, 0, 0])
        ]
        # → 조그 이동량 정의
        # → 한 번 누를 때마다 10mm 이동

        for i, (name, val) in enumerate(jog_dirs):
            tk.Button(
                jog_frame,
                text=name,
                width=8,
                height=2,
                command=lambda v=val: self.send_jog(v)
            ).grid(
                row=i // 2,
                column=i % 2,
                padx=5,
                pady=2
            )
            # → 버튼 클릭 시 JOG 명령 전송


        # ====================================================
        # [4] 좌표 메모리 슬롯
        # ====================================================

        tk.Label(
            root,
            text="[ Memory Slots ]",
            font=("Arial", 10, "bold")
        ).pack(pady=10)

        slot_frame = tk.Frame(root)
        slot_frame.pack()

        for i in range(4):
            tk.Label(
                slot_frame,
                text=f"Slot {i+1}:"
            ).grid(row=i, column=0, padx=5)

            tk.Button(
                slot_frame,
                text="SAVE",
                bg="#e1e1e1",
                command=lambda idx=i: self.save_slot(idx)
            ).grid(row=i, column=1, padx=2, pady=2)
            # → 현재 좌표를 슬롯에 저장

            tk.Button(
                slot_frame,
                text="GOTO",
                bg="#c1c1c1",
                command=lambda idx=i: self.goto_slot(idx)
            ).grid(row=i, column=2, padx=2, pady=2)
            # → 저장된 좌표로 이동


        # ====================================================
        # [5] 상태 표시 바
        # ====================================================

        self.status = tk.Label(
            root,
            text="로봇 연결 대기 중...",
            bd=1,
            relief="sunken",
            anchor="w"
        )
        self.status.pack(side="bottom", fill="x")

        # 서버 시작
        self.start_server()


    # ============================================================
    # TCP 서버 시작
    # ============================================================

    def start_server(self):
        self.server = socket.socket(
            socket.AF_INET,
            socket.SOCK_STREAM
        )
        # → IPv4 + TCP 소켓 생성

        self.server.setsockopt(
            socket.SOL_SOCKET,
            socket.SO_REUSEADDR,
            1
        )
        # → 프로그램 재시작 시 포트 재사용 가능

        self.server.bind((HOST, PORT))
        # → 지정 IP/포트에 바인딩

        self.server.listen(1)
        # → 클라이언트 1개 대기

        self.root.after(100, self.accept_conn)
        # → GUI 멈추지 않게 비동기 연결 대기


    def accept_conn(self):
        self.server.setblocking(False)
        # → accept()가 멈추지 않게 설정

        try:
            self.conn, addr = self.server.accept()
            # → 로봇이 접속하면 소켓 획득

            self.status.config(
                text=f"연결됨: {addr}",
                fg="green"
            )
        except:
            self.root.after(500, self.accept_conn)
            # → 아직 연결 없으면 다시 대기


    # ============================================================
    # 현재 로봇 좌표 요청
    # ============================================================

    def get_pos(self):
        if not self.conn:
            return

        self.conn.sendall(b"GET\n")
        # → 로봇에 현재 좌표 요청

        raw = self.conn.recv(1024).decode().strip()
        # → 로봇에서 응답 수신

        parts = raw.split(',')
        # → "X,Y,Z,Rx,Ry,Rz" 분리

        if len(parts) == 6:
            for i in range(6):
                self.entries[i].delete(0, tk.END)
                self.entries[i].insert(0, parts[i])
                # → GUI에 좌표 표시


    # ============================================================
    # 절대 좌표 이동
    # ============================================================

    def move_abs(self):
        if not self.conn:
            return

        vals = [e.get() for e in self.entries]
        # → Entry에서 값 읽기

        cmd = f"MOVE,{','.join(vals)}\n"
        # → MOVE 명령 생성

        self.conn.sendall(cmd.encode())
        # → 로봇으로 전송


    # ============================================================
    # 조그 이동
    # ============================================================

    def send_jog(self, delta):
        if not self.conn:
            return

        delta_str = [str(d) for d in delta]
        cmd = f"JOG,{','.join(delta_str)}\n"
        self.conn.sendall(cmd.encode())


    # ============================================================
    # 좌표 슬롯 저장
    # ============================================================

    def save_slot(self, idx):
        self.slots[idx] = [e.get() for e in self.entries]
        self.status.config(
            text=f"Slot {idx+1}에 좌표가 저장되었습니다."
        )


    # ============================================================
    # 슬롯 좌표로 이동
    # ============================================================

    def goto_slot(self, idx):
        if self.slots[idx]:
            for i in range(6):
                self.entries[i].delete(0, tk.END)
                self.entries[i].insert(0, self.slots[idx][i])
            self.move_abs()


# ============================================================
# 프로그램 시작 지점
# ============================================================

if __name__ == "__main__":
    root = tk.Tk()
    app = RobotMaster(root)
    root.mainloop()
