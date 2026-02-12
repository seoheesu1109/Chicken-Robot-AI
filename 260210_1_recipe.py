import tkinter as tk
from tkinter import scrolledtext, messagebox
from google import genai
from pydantic import BaseModel
from typing import List
import threading

# 1. 데이터 구조 정의
class Recipe(BaseModel):
    recipe_name: str
    ingredients: List[str]
    steps: List[str]

class RecipeApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Gemini 레시피 AI 추출기")
        self.root.geometry("600x700")

        # API 설정 (여기에 본인의 키를 입력하세요)
        self.api_key = "AIzaSyAIUGSpD5hSnB2bbP_Ng5rkZiQgib-HYUc"
        self.client = genai.Client(api_key=self.api_key)

        self.setup_ui()

    def setup_ui(self):
        # 입력 레이블
        tk.Label(self.root, text="분석할 레시피 텍스트를 입력하세요:", font=('Arial', 10, 'bold')).pack(pady=5)
        
        # 입력창
        self.input_text = scrolledtext.ScrolledText(self.root, height=10)
        self.input_text.pack(padx=10, pady=5, fill=tk.BOTH)
        self.input_text.insert(tk.END, "삼겹살 300g, 김치 한포기 썰어넣고 고추장 한스푼 넣어서 끓여줘.")

        # 분석 버튼
        self.run_btn = tk.Button(self.root, text="AI 레시피 추출 시작", command=self.start_analysis, bg="#4CAF50", fg="white", font=('Arial', 10, 'bold'))
        self.run_btn.pack(pady=10)

        # 결과 레이블
        tk.Label(self.root, text="분석 결과:", font=('Arial', 10, 'bold')).pack(pady=5)

        # 결과 출력창
        self.result_text = scrolledtext.ScrolledText(self.root, height=20, bg="#f0f0f0")
        self.result_text.pack(padx=10, pady=5, fill=tk.BOTH)

    def start_analysis(self):
        # UI 프리징 방지를 위해 스레드 사용
        self.run_btn.config(state=tk.DISABLED, text="분석 중...")
        threading.Thread(target=self.analyze_recipe, daemon=True).start()

    def analyze_recipe(self):
        user_input = self.input_text.get("1.0", tk.END).strip()
        if not user_input:
            messagebox.showwarning("경고", "텍스트를 입력해주세요.")
            self.reset_button()
            return

        try:
            # 아까 확인한 사용 가능한 최신 모델 사용
            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=f"다음 텍스트에서 요리명, 재료, 조리 순서를 정확히 추출해줘: {user_input}",
                config={
                    "response_mime_type": "application/json",
                    "response_schema": Recipe,
                }
            )

            if response.parsed:
                self.display_result(response.parsed)
            else:
                self.result_text.insert(tk.END, "결과를 파싱할 수 없습니다.")

        except Exception as e:
            messagebox.showerror("에러", f"오류 발생: {str(e)}")
        finally:
            self.reset_button()

    def display_result(self, recipe):
        self.result_text.delete("1.0", tk.END)
        res = f"🍴 요리명: {recipe.recipe_name}\n"
        res += "="*40 + "\n"
        res += "🛒 [재료 리스트]\n"
        res += ", ".join(recipe.ingredients) + "\n\n"
        res += "👨‍🍳 [조리 단계]\n"
        for i, step in enumerate(recipe.steps, 1):
            res += f"{i}. {step}\n"
        
        self.result_text.insert(tk.END, res)

    def reset_button(self):
        self.run_btn.config(state=tk.NORMAL, text="AI 레시피 추출 시작")

if __name__ == "__main__":
    root = tk.Tk()
    app = RecipeApp(root)
    root.mainloop()