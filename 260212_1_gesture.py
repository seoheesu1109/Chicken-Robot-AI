import cv2
import mediapipe as mp
import customtkinter as ctk
from PIL import Image, ImageTk
import time
import math

# --- 초기 설정 ---
ctk.set_appearance_mode("dark")
mp_hands = mp.solutions.hands
# 양손 인식을 위해 max_num_hands를 2로 설정
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

class TwoHandGestureApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Chicken Robot Dual Hand Control")
        self.geometry("1000x800")

        self.start_time = 0
        self.hold_duration = 2.0
        self.is_executed = False

        # UI
        ctk.CTkLabel(self, text="🙌 양손 검지를 위로 들어주세요 (2초)", font=("Arial", 24, "bold")).pack(pady=10)
        self.video_label = ctk.CTkLabel(self, text="")
        self.video_label.pack()
        
        self.status_label = ctk.CTkLabel(self, text="대기 중...", font=("Arial", 18), text_color="yellow")
        self.status_label.pack(pady=10)

        self.log_text = ctk.CTkTextbox(self, width=700, height=150)
        self.log_text.pack(pady=20)
        
        self.cap = cv2.VideoCapture(0)
        self.update_frame()

    def is_index_up(self, hand_landmarks):
        """검지만 펴져 있는지 확인하는 함수"""
        # 8번(검지 끝)이 6번(검지 마디)보다 위에 있고, 
        # 나머지 손가락(12, 16, 20)은 각 마디보다 아래에 있는지 확인
        index_up = hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y
        middle_down = hand_landmarks.landmark[12].y > hand_landmarks.landmark[10].y
        ring_down = hand_landmarks.landmark[16].y > hand_landmarks.landmark[14].y
        pinky_down = hand_landmarks.landmark[20].y > hand_landmarks.landmark[18].y
        
        return index_up and middle_down and ring_down and pinky_down

    def update_frame(self):
        success, frame = self.cap.read()
        if success:
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)
            
            both_hands_correct = False

            if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 2:
                # 양손 모두 검지가 위로 향했는지 체크
                hand1 = self.is_index_up(results.multi_hand_landmarks[0])
                hand2 = self.is_index_up(results.multi_hand_landmarks[1])
                
                if hand1 and hand2:
                    both_hands_correct = True
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # --- 로직 처리 ---
            if both_hands_correct:
                if self.start_time == 0:
                    self.start_time = time.time()
                
                elapsed = time.time() - self.start_time
                progress = min(elapsed / self.hold_duration, 1.0)
                
                # 시각적 피드백 (화면 중앙에 큰 게이지)
                cv2.rectangle(frame, (w//2 - 100, h - 50), (w//2 - 100 + int(200 * progress), h - 30), (0, 255, 0), -1)
                self.status_label.configure(text=f"인식 중! ({int(progress*100)}%)", text_color="lime")

                if elapsed >= self.hold_duration and not self.is_executed:
                    self.add_log("🚀 [START] 양손 검지 감지 - 자동조리를 시작합니다!")
                    self.is_executed = True
            else:
                self.start_time = 0
                self.is_executed = False
                self.status_label.configure(text="양손 검지를 올려주세요", text_color="yellow")

            # 영상 표시
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            img_tk = ImageTk.PhotoImage(image=img)
            self.video_label.configure(image=img_tk)
            self.video_label.image = img_tk

        self.after(10, self.update_frame)

    def add_log(self, msg):
        self.log_text.insert("end", f"[{time.strftime('%H:%M:%S')}] {msg}\n")
        self.log_text.see("end")

if __name__ == "__main__":
    app = TwoHandGestureApp()
    app.mainloop()