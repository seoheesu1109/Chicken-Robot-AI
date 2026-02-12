import customtkinter as ctk
import threading
import time

# --- 1. 조리 화면 클래스 (CookingScreen) ---
class CookingScreen(ctk.CTkFrame):
    def __init__(self, master, on_back, **kwargs):
        super().__init__(master, fg_color="white", **kwargs)
        self.on_back = on_back
        self.setup_ui()

    def setup_ui(self):
        # 상단 바 (홈 버튼 + 센서 램프)
        top_bar = ctk.CTkFrame(self, fg_color="transparent")
        top_bar.pack(fill="x", padx=40, pady=(60, 20))
        
        ctk.CTkButton(top_bar, text="🏠 홈으로", width=160, height=60, 
                      font=("Malgun Gothic", 22, "bold"), fg_color="#E5E7EB", 
                      text_color="black", command=self.on_back).pack(side="left")

        # 메인 컨텐츠 영역
        content = ctk.CTkFrame(self, fg_color="transparent")
        content.pack(expand=True, fill="both", padx=40)

        # 왼쪽: 설정 패널
        left_p = ctk.CTkFrame(content, fg_color="transparent", width=400)
        left_p.pack(side="left", fill="y", padx=(0, 20))

        ctk.CTkButton(left_p, text="메뉴 선택 ▾", height=120, font=("Malgun Gothic", 30, "bold")).pack(fill="x", pady=10)
        self.create_card(left_p, "온도 설정", "180°C", "#FB923C")
        self.create_card(left_p, "시간 설정", "30분", "#4ADE80")
        self.create_card(left_p, "쉐이킹 강도", "강도 5", "#C084FC")

        # 오른쪽: 타이머 영역
        right_p = ctk.CTkFrame(content, fg_color="#F3F4F6", corner_radius=40)
        right_p.pack(side="right", expand=True, fill="both")
        
        ctk.CTkLabel(right_p, text="25:00", font=("Arial", 120, "bold"), text_color="#374151").place(relx=0.5, rely=0.4, anchor="center")
        ctk.CTkLabel(right_p, text="남은 조리 시간", font=("Malgun Gothic", 30)).place(relx=0.5, rely=0.55, anchor="center")

        # 하단: 제어 버튼
        bottom_bar = ctk.CTkFrame(self, fg_color="transparent")
        bottom_bar.pack(fill="x", padx=40, pady=60)
        
        btn_data = [("▶ 조리시작", "#22C55E"), ("⏸ 일시정지", "#F59E0B"), ("■ 정지", "#EF4444")]
        for txt, clr in btn_data:
            ctk.CTkButton(bottom_bar, text=txt, height=120, fg_color=clr, 
                          font=("Malgun Gothic", 28, "bold"), corner_radius=20).pack(side="left", expand=True, padx=10)

    def create_card(self, parent, title, value, color):
        card = ctk.CTkFrame(parent, fg_color=color, height=180, corner_radius=25)
        card.pack(fill="x", pady=15)
        ctk.CTkLabel(card, text=title, font=("Malgun Gothic", 18), text_color="white").pack(pady=(20, 0))
        ctk.CTkLabel(card, text=value, font=("Malgun Gothic", 40, "bold"), text_color="white").pack()

# --- 2. 메인 컨트롤 패널 클래스 (MainControlPanel) ---
class MainControlPanel(ctk.CTkFrame):
    def __init__(self, master, on_start_cooking, **kwargs):
        super().__init__(master, fg_color="#F9FAFB", **kwargs)
        
        # 헤더
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=60, pady=(120, 60))
        
        ctk.CTkLabel(header, text="자동 조리 시스템", font=("Malgun Gothic", 50, "bold")).pack(anchor="w")
        ctk.CTkFrame(header, width=150, height=10, fg_color="#3B82F6").pack(anchor="w", pady=15)

        # 메인 버튼들
        btn_container = ctk.CTkFrame(self, fg_color="transparent")
        btn_container.pack(expand=True, fill="x", padx=100)

        # 자동 조리 버튼 클릭 시 조리 화면으로 전환
        ctk.CTkButton(btn_container, text="▶  자동 조리", height=220, corner_radius=40,
                      font=("Malgun Gothic", 45, "bold"), fg_color="#22C55E",
                      command=on_start_cooking).pack(fill="x", pady=20)
        
        ctk.CTkButton(btn_container, text="⚙  설정", height=160, corner_radius=40,
                      font=("Malgun Gothic", 35), fg_color="#E5E7EB", text_color="black").pack(fill="x", pady=20)
        
        ctk.CTkButton(btn_container, text="⏻  종료", height=160, corner_radius=40,
                      font=("Malgun Gothic", 35), fg_color="#374151").pack(fill="x", pady=20)

# --- 3. 앱 관리 클래스 (RobotApp) ---
class RobotApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.geometry("1080x1920")
        self.title("Robot System")
        self.current_screen = None
        
        # 첫 실행: 다이얼로그 띄우기
        self.after(100, self.show_start_dialog)

    def show_start_dialog(self):
        self.dialog = ctk.CTkToplevel(self)
        self.dialog.geometry("800x500+140+710") # 중앙 배치
        self.dialog.overrideredirect(True)
        self.dialog.attributes("-topmost", True)
        
        frame = ctk.CTkFrame(self.dialog, fg_color="white", corner_radius=40, border_width=2, border_color="#E5E7EB")
        frame.pack(fill="both", expand=True)

        ctk.CTkLabel(frame, text="로봇 제어 시스템", font=("Malgun Gothic", 35, "bold")).pack(pady=(60, 20))
        ctk.CTkLabel(frame, text="로봇을 시작하시겠습니까?", font=("Malgun Gothic", 22)).pack(pady=10)

        btn_box = ctk.CTkFrame(frame, fg_color="transparent")
        btn_box.pack(pady=50)

        ctk.CTkButton(btn_box, text="예", width=220, height=90, corner_radius=20,
                      font=("Malgun Gothic", 25, "bold"), fg_color="#22C55E",
                      command=self.start_booting).pack(side="left", padx=15)
        
        ctk.CTkButton(btn_box, text="아니오", width=220, height=90, corner_radius=20,
                      font=("Malgun Gothic", 25, "bold"), fg_color="#6B7280",
                      command=self.quit).pack(side="left", padx=15)

    def start_booting(self):
        self.dialog.destroy()
        # 부팅 오버레이
        self.boot_overlay = ctk.CTkFrame(self, fg_color="#1E1B4B")
        self.boot_overlay.place(relx=0, rely=0, relwidth=1, relheight=1)
        
        ctk.CTkLabel(self.boot_overlay, text="SYSTEM READY", font=("Arial", 60, "bold"), text_color="white").place(relx=0.5, rely=0.45, anchor="center")
        pbar = ctk.CTkProgressBar(self.boot_overlay, width=500)
        pbar.place(relx=0.5, rely=0.55, anchor="center")
        pbar.start()

        threading.Thread(target=self._wait_boot, daemon=True).start()

    def _wait_boot(self):
        time.sleep(2) # 2초 부팅
        self.after(0, self.show_main_screen)

    def clear_screen(self):
        if self.current_screen:
            self.current_screen.destroy()
        if hasattr(self, 'boot_overlay'):
            self.boot_overlay.destroy()

    def show_main_screen(self):
        self.clear_screen()
        self.current_screen = MainControlPanel(self, on_start_cooking=self.show_cooking_screen)
        self.current_screen.pack(fill="both", expand=True)

    def show_cooking_screen(self):
        self.clear_screen()
        self.current_screen = CookingScreen(self, on_back=self.show_main_screen)
        self.current_screen.pack(fill="both", expand=True)

if __name__ == "__main__":
    app = RobotApp()
    app.mainloop()