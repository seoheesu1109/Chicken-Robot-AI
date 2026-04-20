import socket
import cv2
import numpy as np
import threading
import customtkinter as ctk
from PIL import Image, ImageTk
import time
import os
import math # 각도 계산을 위해 추가
import numpy as np

# [1. 설정 데이터 및 경로]
H_MATRIX_PATH = "homography_matrix.npy"
HOME_POSE = "535.65,-246.04,786.67,3.83,124.54,175.95"
DISCHARGE_POSE = "823.70,708.96,962.39,39.90,90.00,170.00"
#SAFE_POSE = "38.25,17.83,97.56,64.79,31.64,92.99"
SAFE_POSE = "770.47,793.10,683.21,73.44,124.54,175.84"
FIXED_ORIENTATION = "74.74,167.13,-13.04" 
MOVE_Z_DEPTH = 828.42 

POT_CENTER_X = 812.0
POT_CENTER_Y = 951.0

X_MIN, X_MAX = 72, 1552
Y_MIN, Y_MAX = 76, 1545

#X_MIN, X_MAX = 472, 1052
#Y_MIN, Y_MAX = 376, 1145

# 구역별 오프셋 데이터 (기존 유지)
AREA_A = [
    {'x': 3.80,  'y': -6.40,  'z': -71.10,  'rx': 77.67, 'ry': 102.78, 'rz': 91.20},
    {'x': 4.70,  'y': 51.80,  'z': -203.20, 'rx': 80.18, 'ry': 112.85, 'rz': 88.94},
    {'x': 62.60, 'y': 97.90,  'z': -183.20, 'rx': 95.37, 'ry': 100.90, 'rz': 96.70},
    {'x': 31.50, 'y': -35.50, 'z': 38.50,   'rx': 89.75, 'ry': 85.50, 'rz': 94.60}
]
AREA_B = [
    {'x': 55.4, 'y': -23.1, 'z': -332.5, 'rx': 98.8, 'ry': 65.4, 'rz': 91.7},
    {'x': 99.8, 'y': -282.4, 'z': -333.8, 'rx': 99.5, 'ry': 64.7, 'rz': 94.8}, 
    {'x': 99.6, 'y': -119.6, 'z': -176.3, 'rx': 97.6, 'ry': 88.5, 'rz': 97.5}, 
    {'x': 40.0,  'y': -105.0, 'z': 110.0,  'rx': 103.0, 'ry': 80.0, 'rz': 92.0}
]
AREA_C = [
    {'x': 29.80,  'y': -0.20,  'z': -123.60, 'rx': 79.50, 'ry': 86.90, 'rz': 129.40},
    {'x': 83.10,  'y': 0.70,   'z': -257.70, 'rx': 79.60, 'ry': 91.10, 'rz': 129.30},
    {'x': 100.80, 'y': -20.90, 'z': -203.30, 'rx': 78.70, 'ry': 88.40, 'rz': 113.50},
    {'x': 66.70,  'y': -38.10, 'z': -61.50,  'rx': 80.10, 'ry': 83.70, 'rz': 101.50}
]

CROP_X, CROP_Y, CROP_W, CROP_H = 190, 100, 670, 550

COLOR_RANGES = {
    "BLUE":   {"low": [90, 100, 100],   "high": [125, 255, 255], "color": (255, 0, 0)},
    "ORANGE": {"low": [5, 120, 100],  "high": [20, 255, 255],  "color": (0, 165, 255)},
    "YELLOW": {"low": [22, 70, 100],  "high": [35, 255, 255],  "color": (0, 255, 255)},
    "PURPLE": {"low": [130, 50, 50],  "high": [165, 255, 255], "color": (255, 0, 255)}
}
#남
SCOOP_SOUTH = [
    {'x': 812.00, 'y': 951.00, 'z': 828.40, 'rx': 74.74,  'ry': 167.13, 'rz': -13.04},
    {'x': 808.47, 'y': 908.94, 'z': 560.36, 'rx': 63.43,  'ry': 166.52, 'rz': -21.52},
    {'x': 798.78, 'y': 766.06, 'z': 485.83, 'rx': 88.05, 'ry':-152.42, 'rz':  -3.81},
    {'x': 802.78, 'y': 750.06, 'z': 571.83, 'rx': 92.26, 'ry':-131.42, 'rz':  -1.81},
    {'x': 814.65, 'y': 811.43, 'z': 570.95, 'rx': 83.66,  'ry':-134.84, 'rz':  1.03},
    {'x': 837.52, 'y': 884.79, 'z': 740.82, 'rx': 84.21,  'ry':-128.83, 'rz':  -0.91},
    {'x': 565.38, 'y': 66.23, 'z': 740.82, 'rx': 84.21,  'ry':-128.83, 'rz':  -0.91},
    {'x': 503.11, 'y': -422.21, 'z': 944.55, 'rx': 91.21,  'ry':150.83, 'rz':  3.81},
    {'x': 637.27, 'y': -562.09, 'z': 911.77, 'rx': 93.71,  'ry':126.48, 'rz':  0.14},
    {'x': 565.38, 'y': 66.23, 'z': 740.82, 'rx': 84.21,  'ry':-128.83, 'rz':  -0.91},
    {'x': 565.38, 'y': 66.23, 'z': 740.82, 'rx': 84.21,  'ry':-128.83, 'rz':  -0.91},
    {'x': 837.52, 'y': 884.79, 'z': 740.82, 'rx': 84.21,  'ry':-128.83, 'rz':  -0.91},
    {'x': 812.00, 'y': 951.00, 'z': 828.40, 'rx': 74.74,  'ry': 167.13, 'rz': -13.04},
]
#동
SCOOP_EAST = [
    {'x': 913.00, 'y': 902.00, 'z': 550.00, 'rx': 158.00, 'ry': 166.00, 'rz': -20.00},
    {'x': 917.00, 'y': 905.00, 'z': 476.00, 'rx':  19.60, 'ry': 149.89, 'rz':-170.00},
    {'x': 936.78, 'y': 904.06, 'z': 560.83, 'rx': 1.71, 'ry':129.42, 'rz':  -178.81},
    {'x': 764.00, 'y': 856.00, 'z': 710.00, 'rx':   6.35, 'ry': 130.00, 'rz':-172.00},
    {'x': 565.38, 'y': 66.23, 'z': 740.82, 'rx': 84.21,  'ry':-128.83, 'rz':  -0.91},
    {'x': 503.11, 'y': -422.21, 'z': 944.55, 'rx': 91.21,  'ry':150.83, 'rz':  3.81},
    {'x': 637.27, 'y': -562.09, 'z': 911.77, 'rx': 93.71,  'ry':126.48, 'rz':  0.14},
    {'x': 565.38, 'y': 66.23, 'z': 740.82, 'rx': 84.21,  'ry':-128.83, 'rz':  -0.91},
    {'x': 565.38, 'y': 66.23, 'z': 740.82, 'rx': 84.21,  'ry':-128.83, 'rz':  -0.91},
    {'x': 837.52, 'y': 884.79, 'z': 740.82, 'rx': 84.21,  'ry':-128.83, 'rz':  -0.91},
    {'x': 812.00, 'y': 951.00, 'z': 828.40, 'rx': 74.74,  'ry': 167.13, 'rz': -13.04},
]
#서
SCOOP_WEST = [
    {'x': 889.00, 'y': 868.00, 'z': 550.00, 'rx':  33.00, 'ry': 159.00, 'rz':  30.00},
    {'x': 626.00, 'y': 910.00, 'z': 498.00, 'rx': 179.00, 'ry': 166.00, 'rz': 179.58},
    {'x': 697.99, 'y': 917.93, 'z': 511.90, 'rx': 176.00, 'ry': 123.00, 'rz': -176.00},
    {'x': 754.99, 'y': 838.00, 'z': 698.00, 'rx': 170.00, 'ry': 121.00, 'rz': -176.00},
    {'x': 630.00, 'y': 249.80, 'z': 711.00, 'rx': 80.34, 'ry': -126.17, 'rz': 3.44},
    {'x': 565.38, 'y': 66.23, 'z': 740.82, 'rx': 84.21,  'ry':-128.83, 'rz':  -0.91},
    {'x': 503.11, 'y': -422.21, 'z': 944.55, 'rx': 91.21,  'ry':150.83, 'rz':  3.81},
    {'x': 637.27, 'y': -562.09, 'z': 911.77, 'rx': 93.71,  'ry':126.48, 'rz':  0.14},
    {'x': 565.38, 'y': 66.23, 'z': 740.82, 'rx': 84.21,  'ry':-128.83, 'rz':  -0.91},
    {'x': 565.38, 'y': 66.23, 'z': 740.82, 'rx': 84.21,  'ry':-128.83, 'rz':  -0.91},
    {'x': 837.52, 'y': 884.79, 'z': 740.82, 'rx': 84.21,  'ry':-128.83, 'rz':  -0.91},
    {'x': 812.00, 'y': 951.00, 'z': 828.40, 'rx': 74.74,  'ry': 167.13, 'rz': -13.04},
]
#북
SCOOP_NORTH = [
    {'x': 796.00, 'y': 772.77, 'z': 552.00, 'rx': 123.00, 'ry': -173.00, 'rz':-149.00},
    {'x': 767.00, 'y': 1044.00, 'z': 509.78, 'rx':  80.43, 'ry': 150.98, 'rz':171.00},
    {'x': 811.99, 'y': 998.66, 'z': 516.25, 'rx': 85.22, 'ry': 119.36, 'rz': 178.00},
    {'x': 766.46, 'y': 914.60, 'z': 688.00, 'rx':  86.48, 'ry': 123.00, 'rz':177.00},
    {'x': 688.32, 'y': 890.92, 'z': 670.30, 'rx': 1.21, 'ry': -121.45, 'rz':0.50},
    {'x': 565.38, 'y': 66.23, 'z': 740.82, 'rx': 84.21,  'ry':-128.83, 'rz':  -0.91},
    {'x': 503.11, 'y': -422.21, 'z': 944.55, 'rx': 91.21,  'ry':150.83, 'rz':  3.81},
    {'x': 637.27, 'y': -562.09, 'z': 911.77, 'rx': 93.71,  'ry':126.48, 'rz':  0.14},
    {'x': 565.38, 'y': 66.23, 'z': 740.82, 'rx': 84.21,  'ry':-128.83, 'rz':  -0.91},
    {'x': 565.38, 'y': 66.23, 'z': 740.82, 'rx': 84.21,  'ry':-128.83, 'rz':  -0.91},
    {'x': 837.52, 'y': 884.79, 'z': 740.82, 'rx': 84.21,  'ry':-128.83, 'rz':  -0.91},
    {'x': 812.00, 'y': 951.00, 'z': 828.40, 'rx': 74.74,  'ry': 167.13, 'rz': -13.04},
]
#북서
SCOOP_NORTHWEST =  [
    {'x': 855.00, 'y': 881.37, 'z': 569.75, 'rx': 158.50,  'ry':-164.83, 'rz': -159.84},
    {'x': 710.22, 'y': 1011.84, 'z': 494.56, 'rx': 129.43,  'ry': 158.52, 'rz': 178.12},
    {'x': 733.38, 'y': 996.80, 'z': 549.50, 'rx': 128.40, 'ry': 126.40, 'rz': -179.50},
    {'x': 762.31, 'y': 966.92, 'z': 643.99, 'rx':128.42, 'ry':120.89, 'rz':  176.81},
    {'x': 630.00, 'y': 249.80, 'z': 711.00, 'rx': 80.34, 'ry': -126.17, 'rz': 3.44},
    {'x': 565.38, 'y': 66.23, 'z': 740.82, 'rx': 84.21,  'ry':-128.83, 'rz':  -0.91},
    {'x': 503.11, 'y': -422.21, 'z': 944.55, 'rx': 91.21,  'ry':150.83, 'rz':  3.81},
    {'x': 637.27, 'y': -562.09, 'z': 911.77, 'rx': 93.71,  'ry':126.48, 'rz':  0.14},
    {'x': 565.38, 'y': 66.23, 'z': 740.82, 'rx': 84.21,  'ry':-128.83, 'rz':  -0.91},
    {'x': 565.38, 'y': 66.23, 'z': 740.82, 'rx': 84.21,  'ry':-128.83, 'rz':  -0.91},
    {'x': 837.52, 'y': 884.79, 'z': 740.82, 'rx': 84.21,  'ry':-128.83, 'rz':  -0.91},
    {'x': 812.00, 'y': 951.00, 'z': 828.40, 'rx': 74.74,  'ry': 167.13, 'rz': -13.04},
]
#남서
SCOOP_SOUTHWEST = [
    {'x': 902.16, 'y': 964.14, 'z': 538.80, 'rx': 73.35,  'ry': 171.32, 'rz': 45.54},
    {'x': 728.79, 'y': 822.48, 'z': 461.53, 'rx': 28.95,  'ry':-144.52, 'rz': -9.38},
    {'x': 732.09, 'y': 805.00, 'z': 507.00, 'rx': 33.08, 'ry': -119.40, 'rz': -4.90},
    {'x': 772.96, 'y': 801.22, 'z': 692.02, 'rx': 27.05, 'ry':-118.12, 'rz':  -6.81},
    {'x': 565.38, 'y': 66.23, 'z': 740.82, 'rx': 84.21,  'ry':-128.83, 'rz':  -0.91},
    {'x': 503.11, 'y': -422.21, 'z': 944.55, 'rx': 91.21,  'ry':150.83, 'rz':  3.81},
    {'x': 637.27, 'y': -562.09, 'z': 911.77, 'rx': 93.71,  'ry':126.48, 'rz':  0.14},
    {'x': 565.38, 'y': 66.23, 'z': 740.82, 'rx': 84.21,  'ry':-128.83, 'rz':  -0.91},
    {'x': 565.38, 'y': 66.23, 'z': 740.82, 'rx': 84.21,  'ry':-128.83, 'rz':  -0.91},
    {'x': 837.52, 'y': 884.79, 'z': 740.82, 'rx': 84.21,  'ry':-128.83, 'rz':  -0.91},
    {'x': 812.00, 'y': 951.00, 'z': 828.40, 'rx': 74.74,  'ry': 167.13, 'rz': -13.04},
]
#남동
SCOOP_SOUTHEAST = [
    {'x': 701.51, 'y':1036.77, 'z': 533.26, 'rx':  122.74, 'ry': 174.11, 'rz': -11.62},
    {'x': 905.30, 'y': 822.00, 'z': 486.23, 'rx': 141.62, 'ry': -150.45, 'rz': 3.29},
    {'x': 906.99, 'y': 807.00, 'z': 586.60, 'rx': 141.50, 'ry': -127.00, 'rz': 2.80},
    {'x': 842.09, 'y': 874.86, 'z': 698.71, 'rx': 146.58, 'ry': -114.68, 'rz': 0.16},
    {'x': 565.38, 'y': 66.23, 'z': 740.82, 'rx': 84.21,  'ry':-128.83, 'rz':  -0.91},
    {'x': 503.11, 'y': -422.21, 'z': 944.55, 'rx': 91.21,  'ry':150.83, 'rz':  3.81},
    {'x': 637.27, 'y': -562.09, 'z': 911.77, 'rx': 93.71,  'ry':126.48, 'rz':  0.14},
    {'x': 565.38, 'y': 66.23, 'z': 740.82, 'rx': 84.21,  'ry':-128.83, 'rz':  -0.91},
    {'x': 565.38, 'y': 66.23, 'z': 740.82, 'rx': 84.21,  'ry':-128.83, 'rz':  -0.91},
    {'x': 837.52, 'y': 884.79, 'z': 740.82, 'rx': 84.21,  'ry':-128.83, 'rz':  -0.91},
    {'x': 812.00, 'y': 951.00, 'z': 828.40, 'rx': 74.74,  'ry': 167.13, 'rz': -13.04},
]
#북동
SCOOP_NORTHEAST = [
    {'x': 753.63, 'y': 839.00, 'z': 502.00, 'rx': 16.00, 'ry':-168.00, 'rz':148.75},
    {'x': 912.00, 'y': 1002.00, 'z': 509.14, 'rx':  54.76, 'ry': 153.08, 'rz':-166.79},
    {'x': 887.99, 'y': 974.00, 'z': 548.60, 'rx': 46.90, 'ry': 121.20, 'rz': -179.00},
    {'x': 915.46, 'y': 970.60, 'z': 725.35, 'rx':  48.09, 'ry': 127.00, 'rz':178.74},
    {'x': 797.92, 'y': 759.60, 'z': 730.00, 'rx': 147.32, 'ry': -114.45, 'rz':3.50},
    {'x': 565.38, 'y': 66.23, 'z': 740.82, 'rx': 84.21,  'ry':-128.83, 'rz':  -0.91},
    {'x': 503.11, 'y': -422.21, 'z': 944.55, 'rx': 91.21,  'ry':150.83, 'rz':  3.81},
    {'x': 637.27, 'y': -562.09, 'z': 911.77, 'rx': 93.71,  'ry':126.48, 'rz':  0.14},
    {'x': 565.38, 'y': 66.23, 'z': 740.82, 'rx': 84.21,  'ry':-128.83, 'rz':  -0.91},
    {'x': 565.38, 'y': 66.23, 'z': 740.82, 'rx': 84.21,  'ry':-128.83, 'rz':  -0.91},
    {'x': 837.52, 'y': 884.79, 'z': 740.82, 'rx': 84.21,  'ry':-128.83, 'rz':  -0.91},
    {'x': 812.00, 'y': 951.00, 'z': 828.40, 'rx': 74.74,  'ry': 167.13, 'rz': -13.04},
]
class RobotApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Doosan Robot Kitchen - Stable Control System")
        self.geometry("1400x900")
        
        # [1] 데이터 로드 (지우지 마세요! 꼭 필요합니다)
        self.load_homography() 
        
        # [2] 변수 초기화
        self.last_pixel_pos = None 
        self.last_target_pos = None
        self.conn = None
        self.cap = None
        self.last_target_pos = None
        self.current_assigned_pos = None
        self.target_color_key = "BLUE"
        self.is_moving = False
        self.is_all_auto = False
        self.stop_requested = False

        # [3] UI 구성 (이 안에서 스크롤 사이드바를 만듭니다)
        self.setup_ui()
        
        # [4] 나머지 설정 실행
        self.setup_camera()
        self.start_socket_server()
        
    def load_homography(self):
        global H_matrix
        if os.path.exists(H_MATRIX_PATH):
            H_matrix = np.load(H_MATRIX_PATH)
        else:
            H_matrix = None

    def setup_ui(self):
        self.grid_columnconfigure(0, minsize=300) # 사이드바 너비 고정
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.sidebar = ctk.CTkScrollableFrame(self, width=280, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        
        ctk.CTkLabel(self.sidebar, text="🤖 KITCHEN MASTER", font=("Arial", 22, "bold")).pack(pady=20)

        btn_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        btn_frame.pack(pady=10, padx=20, fill="x")
        ctk.CTkButton(btn_frame, text="🏠 HOME", fg_color="#2980b9", font=("Arial", 14, "bold"), 
                      height=45, command=self.execute_home_move).pack(side="left", expand=True, padx=5)
        ctk.CTkButton(btn_frame, text="🛡 SAFE", fg_color="#27ae60", font=("Arial", 12, "bold"), 
                      height=40, command=self.execute_safe_move).pack(side="left", expand=True, padx=2)
        ctk.CTkButton(btn_frame, text="📤 DISCHARGE", fg_color="#8e44ad", font=("Arial", 14, "bold"), 
                      height=45, command=self.execute_discharge_move).pack(side="left", expand=True, padx=5)

        ctk.CTkLabel(self.sidebar, text="--- SEQUENCES ---", font=("Arial", 12)).pack(pady=5)
        self.discharge_seq_btn = ctk.CTkButton(
            self.sidebar, text="🔥 DISCHARGE SEQ", fg_color="#27ae60", font=("Arial", 16, "bold"), 
            height=50, command=self.execute_discharge_sequence
        )
        self.discharge_seq_btn.pack(pady=5, padx=20, fill="x")

        # [추가] SCREW 시퀀스 버튼
        self.screw_btn = ctk.CTkButton(
            self.sidebar, text="🌀 SCREW (STIR)", fg_color="#16a085", font=("Arial", 16, "bold"), 
            height=50, command=self.execute_screw_move
        )
        self.call_sub_btn = ctk.CTkButton(self.sidebar, text="⚡ ddischarge", fg_color="#e67e22", 
                                         font=("Arial", 14, "bold"), height=45, command=self.execute_sub_program_call)
        self.call_sub_btn.pack(pady=3, padx=20, fill="x")
        ctk.CTkButton(self.sidebar, text="📥 BADE INPUT", fg_color="#34495e", 
                      font=("Arial", 14, "bold"), height=45, 
                      command=lambda: self.execute_named_sub_program("badeinput")).pack(pady=3, padx=20, fill="x")

        ctk.CTkButton(self.sidebar, text="🔥 FRY BADE", fg_color="#c0392b", 
                      font=("Arial", 14, "bold"), height=45, 
                      command=lambda: self.execute_named_sub_program("frybade")).pack(pady=3, padx=20, fill="x")

        ctk.CTkButton(self.sidebar, text="📤 FRY OUTPUT", fg_color="#2980b9", 
                      font=("Arial", 14, "bold"), height=45, 
                      command=lambda: self.execute_named_sub_program("fryoutput")).pack(pady=3, padx=20, fill="x")
        ctk.CTkLabel(self.sidebar, text="--- COLOR MODE ---", font=("Arial", 12)).pack(pady=5)
        for color in COLOR_RANGES.keys():
            btn_color = "#3498db" if color=="BLUE" else "#e67e22" if color=="ORANGE" else "#f1c40f" if color=="YELLOW" else "#9b59b6"
            ctk.CTkButton(self.sidebar, text=f"{color} MODE", fg_color=btn_color, 
                          text_color="white" if color != "YELLOW" else "black",
                          command=lambda c=color: self.set_target_color(c)).pack(pady=3, padx=20, fill="x")

        ctk.CTkLabel(self.sidebar, text="--- ACTIONS ---", font=("Arial", 12)).pack(pady=10)
        ctk.CTkButton(self.sidebar, text="🚀 TRACK MOVE", fg_color="#27ae60", font=("Arial", 16, "bold"), 
                      height=50, command=self.execute_marker_move).pack(pady=5, padx=20, fill="x")
        ctk.CTkButton(self.sidebar, text="🔨 TRIPLE STRIKE", fg_color="#d35400", font=("Arial", 16, "bold"), 
                      height=60, command=self.execute_strike_move).pack(pady=10, padx=20, fill="x")
        ctk.CTkButton(self.sidebar, text="🥄 DUAL SCOOP", fg_color="#c0392b", font=("Arial", 16, "bold"), 
                      height=50, command=self.execute_dual_scoop).pack(pady=5, padx=20, fill="x")
        ctk.CTkLabel(self.sidebar, text="--- ACTIONS ---", font=("Arial", 12)).pack(pady=10)
        ctk.CTkButton(self.sidebar, text="📍 GO TO POT CENTER", fg_color="#2ecc71", font=("Arial", 16, "bold"), 
                      height=50, command=self.execute_marker_move).pack(pady=5, padx=20, fill="x")

        # 2. 거기서 공 각도에 맞춰 손목만 돌리기
        ctk.CTkButton(self.sidebar, text="🔄 ADJUST RZ ONLY", fg_color="#3498db", font=("Arial", 16, "bold"), 
                      height=50, command=self.execute_rz_adjustment).pack(pady=5, padx=20, fill="x")
        ctk.CTkButton(self.sidebar, text="🥄 SCOOP MOTION", fg_color="#1abc9c", font=("Arial", 16, "bold"), 
              height=50, command=self.execute_scoop_motion).pack(pady=5, padx=20, fill="x")
       # 1. 고정 방향으로 이동하는 버튼
        self.status_label = ctk.CTkLabel(self.sidebar, text="Offline", text_color="#E67E22", font=("Arial", 14))
        self.status_label.pack(side="bottom", pady=20)

        self.video_container = ctk.CTkFrame(self, fg_color="#121212")
        self.video_container.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        self.video_label = ctk.CTkLabel(self.video_container, text="")
        self.video_label.pack(expand=True)
        self.all_auto_btn = ctk.CTkButton(
        self.sidebar, text="🤖 ALL-AUTO START", fg_color="#e74c3c", font=("Arial", 18, "bold"), 
        height=60, command=self.toggle_all_auto
        )
        self.all_auto_btn.pack(pady=15, padx=20, fill="x")
    # 카메라 및 서버 설정 (기존 유지)
    def setup_camera(self):
        if self.cap is not None: self.cap.release()
        for idx in [1, 0, 2]:
            self.cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            if self.cap.isOpened(): break
        if self.cap.isOpened():
            self.cap.set(3, 1280); self.cap.set(4, 720)
            self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1); #self.cap.set(cv2.CAP_PROP_FOCUS, 1)
            self.update_video()

    def update_video(self):
        if self.cap is None or not self.cap.isOpened(): return
        ret, frame = self.cap.read()
        if ret:
            display_frame = frame.copy()
            center_x, center_y = 550, 350  # 냄비 중심점
            
            # --- [여기부터 부채꼴 격자 그리기 추가] ---
            # 1. 방사형 선 (각도 분할: 60도씩 6등분)
            for i in range(18):
                # i=0일 때 -180도부터 시작해서 20도씩 증가
                angle_deg = i * 20 - 180  
                angle_rad = math.radians(angle_deg)
                
                # 선의 길이를 냄비 끝까지 충분히 길게 설정 (600픽셀)
                line_len = 600
                end_x = int(center_x + line_len * math.cos(angle_rad))
                end_y = int(center_y - line_len * math.sin(angle_rad))
                
                # 파란색 선으로 각도 영역 표시 (두께 1)
                cv2.line(display_frame, (center_x, center_y), (end_x, end_y), (255, 100, 0), 1)
                mid_angle_deg = angle_deg + 10 
                mid_angle_rad = math.radians(mid_angle_deg)
                
                # 번호를 표시할 위치 (중심에서 350픽셀 떨어진 지점)
                text_dist = 350 
                txt_x = int(center_x + text_dist * math.cos(mid_angle_rad))
                txt_y = int(center_y - text_dist * math.sin(mid_angle_rad))
                
                # 구역 번호 써주기 (노란색으로 작게 표시)
                cv2.putText(display_frame, f"[{i}]", (txt_x - 10, txt_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                # 각도 라벨 표시 (선 끝부분에 작게 표시)
                # 가독성을 위해 40도 간격으로만 텍스트를 써줘도 좋습니다.
                if i % 2 == 0:
                    cv2.putText(display_frame, f"{angle_deg}d", (end_x, end_y), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 150, 50), 1)

            # 2. 중심점 마커 (빨간 십자)
            cv2.drawMarker(display_frame, (center_x, center_y), (0, 0, 255), 
                           cv2.MARKER_CROSS, 30, 2)

            # 3. 중심점 마커
            cv2.drawMarker(display_frame, (center_x, center_y), (0, 0, 255), 
                           cv2.MARKER_CROSS, 30, 2)
            # --- [격자 그리기 끝] ---

            # 기존 로직 (CROP 및 색상 인식)
            cropped = frame[CROP_Y:CROP_Y+CROP_H, CROP_X:CROP_X+CROP_W]
            hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
            
            # ... (중략: mask 생성 및 컨투어 검출 로직) ...

            # 물체를 찾았을 때 화면에 현재 위치 인덱스 표시
            if hasattr(self, 'last_pixel_pos') and self.last_pixel_pos:
                bx, by = self.last_pixel_pos
                # 현재 물체가 몇 번 칸에 있는지 계산 (티칭 보조용)
                dist = math.sqrt((bx-center_x)**2 + (by-center_y)**2)
                ang = math.degrees(math.atan2(-(by-center_y), bx-center_x))
                
                r_idx = int(np.clip(dist / (max_r/6), 0, 5))
                a_idx = int((ang + 180) / 60) % 6
                
                # 화면 좌측 상단에 현재 구역 정보 표시
                cv2.putText(display_frame, f"Current Sector: R[{r_idx}] A[{a_idx}]", (20, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # 1. 모든 색상 통합 마스크
            combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for color_key, cfg in COLOR_RANGES.items():
                mask = cv2.inRange(hsv, np.array(cfg["low"]), np.array(cfg["high"]))
                combined_mask = cv2.bitwise_or(combined_mask, mask)
            
            combined_mask = cv2.dilate(cv2.erode(combined_mask, None, iterations=2), None, iterations=2)
            contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # 임시 리스트: (로봇좌표_x, 로봇좌표_y, 면적, 화면좌표_x, 화면좌표_y, 반지름)
            temp_targets = []
            
            for c in contours:
                area = cv2.contourArea(c) # 컨투어 면적 계산
                if area > 500: # 최소 면적 기준 (노이즈 제거)
                    ((x, y), radius) = cv2.minEnclosingCircle(c)
                    full_cx = int(x) + CROP_X
                    full_cy = int(y) + CROP_Y
                    
                    cx_raw = (full_cx - 640) * (500 / 900)
                    cy_raw = (full_cy - 360) * (500 / 900)
                    
                    if H_matrix is not None:
                        input_pt = np.array([[[cx_raw, cy_raw]]], dtype=np.float32)
                        robot_pt = cv2.perspectiveTransform(input_pt, H_matrix)
                        rx, ry = robot_pt[0][0][0], robot_pt[0][0][1]
                        
                        temp_targets.append({
                            'pos': [rx, ry],
                            'area': area,
                            'ui_pos': (full_cx, full_cy),
                            'r': int(radius)
                        })

            # 2. 면적(area) 기준으로 내림차순 정렬 (가장 큰 것이 0번 인덱스)
            temp_targets.sort(key=lambda x: x['area'], reverse=True)
            
            # 정제된 로봇 좌표 리스트 업데이트
            self.all_detected_targets = [t['pos'] for t in temp_targets]
            
            # 3. 화면 표시 (가장 큰 것만 강조하거나 순번 표시)
            for i, target in enumerate(temp_targets):
                color = (0, 255, 0) if i == 0 else (255, 255, 255) # 1순위는 녹색, 나머지는 흰색
                thickness = 3 if i == 0 else 1
                
                cv2.circle(display_frame, target['ui_pos'], target['r'], color, thickness)
                cv2.putText(display_frame, f"#{i+1} Area:{int(target['area'])}", 
                            (target['ui_pos'][0]-10, target['ui_pos'][1]-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # 자동 모드에서 참조할 타겟 (가장 큰 물체)
            if self.all_detected_targets:
                self.last_target_pos = self.all_detected_targets[0]
            else:
                self.last_target_pos = None

            # UI 영상 업데이트 로직 (동일)
            cv2.rectangle(display_frame, (CROP_X, CROP_Y), (CROP_X+CROP_W, CROP_Y+CROP_H), (255, 255, 255), 2)
            img = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB); img_pil = Image.fromarray(img)
            img_tk = ctk.CTkImage(light_image=img_pil, dark_image=img_pil, size=(900, 500))
            self.video_label.configure(image=img_tk)
            
        self.after(20, self.update_video)

    def start_socket_server(self):
        def server_thread():
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("0.0.0.0", 30002)); s.listen(1)
            while True:
                conn, addr = s.accept(); self.conn = conn
                self.status_label.configure(text=f"✅ Robot Connected", text_color="#1ABC9C")
        threading.Thread(target=server_thread, daemon=True).start()

    def send_command_and_wait(self, cmd):
        if not self.conn: return False
        try:
            self.conn.sendall((cmd + "\n").encode())
            self.conn.settimeout(25.0)
            return "DONE" in self.conn.recv(1024).decode()
        except: return False

    def _move_worker(self, mode, target_pos_at_click):
        if self.is_moving: return
        self.is_moving = True

        bx, by = None, None
        # target_pos_at_click 처리 (좌표 리스트나 단일 좌표 대응)
        if target_pos_at_click:
            if isinstance(target_pos_at_click, (list, tuple)):
                bx = np.clip(target_pos_at_click[0], X_MIN, X_MAX)
                by = np.clip(target_pos_at_click[1], Y_MIN, Y_MAX)
            else:
                bx = np.clip(target_pos_at_click, X_MIN, X_MAX)

        
        if mode == "HOME":
            self.status_label.configure(text="🏠 Moving Home...", text_color="#3498db")
            if self.send_command_and_wait(f"MOVE,{HOME_POSE}"):
                self.current_assigned_pos = None

        elif mode == "SAFE":
            self.status_label.configure(text="🛡 Moving to Safe Pose...", text_color="#27ae60")
            result = self.send_command_and_wait(f"MOVE,{SAFE_POSE}")
            print(f"SAFE 결과: {result}")
            if result:
                self.current_assigned_pos = None

        # [추가] 센터 이동 버튼용 (인자 4개를 풀어서 전송)
        elif mode == "FIXED_POS" and target_pos_at_click:
            try:
                fx, fy, fz, frot = target_pos_at_click
                self.status_label.configure(text="🚀 Moving to POT CENTER", text_color="#27ae60")
                self.send_command_and_wait(f"MOVE,{fx:.2f},{fy:.2f},{fz:.2f},{frot}")
            except: pass

        elif mode == "DISCHARGE":
            self.status_label.configure(text="📤 Discharging...", text_color="#8e44ad")
            if self.send_command_and_wait(f"MOVE,{DISCHARGE_POSE}"):
                self.current_assigned_pos = None

        elif mode in ["ddischarge", "badeinput", "frybade", "fryoutput"]:
            self.status_label.configure(text=f"📤 Executing {mode}...", text_color="#8e44ad")
            if self.send_command_and_wait(f"{mode}"):
                self.current_assigned_pos = None

        elif mode == "DISCHARGE_SEQ":
            self.status_label.configure(text="🚀 Running Discharge Seq...", text_color="#2ecc71")
            discharge_path = ["690,537,685,90,-124,1,200,50", "577,88,672,82,-139,0.18,200,50", "515,-341,890,78.156,-9,200,50","670,94,723,70,104,-174,200,50"]
            for i, pos in enumerate(discharge_path):
                if self.send_command_and_wait(f"MOVE,{pos}"):
                    time.sleep(0.5)
                    if i == 2: time.sleep(2)
                else: break
            self.current_assigned_pos = None

        elif mode == "SCREW":
            self.status_label.configure(text="🌀 Screwing (3 Rounds)...", text_color="#1ABC9C")
            screw_vel, screw_res = 650, 300
            screw_points = ["1006,698,832", "745,1029,832", "613,781,871", "783,487,879"]
            fixed_rot = "96,87,100"
            for r in range(3):
                for pt in screw_points:
                    if not self.send_command_and_wait(f"MOVE,{pt},{fixed_rot},{screw_vel},{screw_res}"): break
                    time.sleep(0.5)

        elif mode == "STRIKE" and bx is not None:
            self.status_label.configure(text="🔨 Fast Triple Striking...", text_color="#d35400")
            strike_vel, strike_r = 1000, 3
            for i in range(4):
                self.send_command_and_wait(f"MOVE,{bx:.2f},{by:.2f},877.00,{FIXED_ORIENTATION},{strike_vel},{strike_r}")
                self.send_command_and_wait(f"MOVE,{bx:.2f},{by:.2f},950.00,{FIXED_ORIENTATION},{strike_vel},{strike_r}")

        elif mode == "TRACK" and bx is not None:
            self.status_label.configure(text="🚀 Moving to Fixed Pose", text_color="#27ae60")
            self.send_command_and_wait(f"MOVE,{bx:.2f},{by:.2f},{MOVE_Z_DEPTH:.2f},{FIXED_ORIENTATION}")

        # [추가] RZ 자동 계산 이동 버튼용
        # [추가] RZ 자동 계산 이동 버튼용
        elif mode == "ADJUST_RZ" and bx is not None:
            # 1. 현재 계산된 J6 각도 (예: 370.0)
            dynamic_j6 = self.calculate_polar_j6(bx, by) 

            # 2. [핵심 수정] 360도 나머지 연산을 통해 항상 -180 ~ +180 사이로 정규화
            # (dynamic_j6 + 180) % 360 - 180 공식은 370도를 10도로, -190도를 170도로 바꿔줍니다.
            safe_target_j6 = (dynamic_j6 + 180) % 360 - 180
            
            # 3. 로봇 회전값 설정 (안전한 각도인 safe_target_j6 사용)
            new_rot = f"74.74,167.13,{safe_target_j6:.2f}"
            
            # UI 표시 업데이트
            self.status_label.configure(text=f"🔄 J6: {safe_target_j6:.1f}°", text_color="#3498db")
            
            # 4. 로봇에게 최종 명령 전송
            cmd = f"MOVE,812.00,951.00,828.42,{new_rot}"
            print(f"📡 Robot Command: {cmd} (Original was: {dynamic_j6:.2f})") 
            self.send_command_and_wait(cmd)

        # _move_worker 안의 SCOOP_MOTION 분기를 아래로 교체

        elif mode == "SCOOP_MOTION" and bx is not None:
            # 실측 픽셀 앵커 (직접 취득한 데이터)
            DIRECTION_ANCHORS = {
                "남":  (704.79, 363.97),
                "서":  (281.48, 768.26),
                "북":  (585.68, 1135.82),
                "동":  (1050.64, 808.60),
                "북서": (327.19, 957.74),
                "북동": (900.30, 1044.96),
                "남동": (981.52, 527.08),
                "남서": (445.21, 443.99),
            }
            SCOOP_MAP = {
                "남":  SCOOP_SOUTH,
                "서":  SCOOP_WEST,
                "북":  SCOOP_NORTH,
                "동":  SCOOP_EAST,
                "북서": SCOOP_NORTHWEST,
                "북동": SCOOP_NORTHEAST,
                "남동": SCOOP_SOUTHEAST,
                "남서": SCOOP_SOUTHWEST,
            }

            # 유클리드 거리로 가장 가까운 방향 선택
            direction = min(
                DIRECTION_ANCHORS,
                key=lambda d: math.hypot(bx - DIRECTION_ANCHORS[d][0],
                                        by - DIRECTION_ANCHORS[d][1])
            )
            selected = SCOOP_MAP[direction]

            self.status_label.configure(text=f"🥄 Scooping {direction}...", text_color="#1abc9c")
            print(f"🥄 방향: {direction} | Pixel: ({bx:.1f}, {by:.1f})")

            for i, pose in enumerate(selected):
                blending = 0 if i == len(selected) - 1 else 50
                cmd = f"MOVE,{pose['x']:.2f},{pose['y']:.2f},{pose['z']:.2f},{pose['rx']:.2f},{pose['ry']:.2f},{pose['rz']:.2f},400,{blending}"
                print(f"  Step {i+1}: {cmd}")
                if not self.send_command_and_wait(cmd):
                    print(f"  ❌ Step {i+1} failed")
                    break
                time.sleep(0.1)

            self.current_assigned_pos = None
        self.status_label.configure(text="✅ Complete", text_color="#1ABC9C")
        self.is_moving = False
    def toggle_all_auto(self):
        if not self.is_all_auto:
            self.is_all_auto = True
            self.stop_requested = False
            self.all_auto_btn.configure(text="🛑 STOP AFTER SESSION", fg_color="#c0392b")
            threading.Thread(target=self._all_auto_worker, daemon=True).start()
        else:
            self.stop_requested = True
            self.status_label.configure(text="⏳ Stopping after this session...", text_color="#f1c40f")

    def _all_auto_worker(self):
        while self.is_all_auto:
            try:
                # 1. 탐지된 물체 리스트 확인
                targets = list(self.all_detected_targets) if hasattr(self, 'all_detected_targets') else []
                
                if not targets:
                    self.status_label.configure(text="🔍 Searching for ANY object...", text_color="#3498db")
                    time.sleep(0.5)
                    continue

                # 2. 가장 가까운(또는 리스트의 첫 번째) 물체 선택
                target_pos = targets[0] 
                bx = np.clip(target_pos[0], X_MIN, X_MAX)
                by = np.clip(target_pos[1], Y_MIN, Y_MAX)

                # 3. 작업 속도 향상을 위해 blending과 속도 최적화
                # 트래킹 이동
                self.status_label.configure(text="🚀 Targeting...", text_color="#2ecc71")
                if not self.send_command_and_wait(f"MOVE,{bx:.2f},{by:.2f},{MOVE_Z_DEPTH:.2f},{FIXED_ORIENTATION},600,50"): break

                # 4. 스쿱(건져내기) 실행
                self._execute_scoop_logic(bx, by)

                # 5. 배출 시퀀스 (속도 대폭 상향 및 정지 최소화)
                self.status_label.configure(text="📤 Fast Discharge", text_color="#8e44ad")
                discharge_path = [
                    "670,758,1090,78,85,92,400,50",
                    "975,12,1163.9,18.34,89.07,97.27,400,50",
                    "999,-372,777,2.24,77.20,175.45", # 배출 지점은 정지
                    "943,30,1164,43.3,84,101.8,400,0"
                ]
                for pos in discharge_path:
                    self.send_command_and_wait(f"MOVE,{pos}")
                
                # 6. 세션 종료 체크
                if self.stop_requested:
                    break

            except Exception as e:
                print(f"Auto Error: {e}")
                break
        
        # 루프 종료 시 안전 위치 복귀 및 버튼 초기화
        self.send_command_and_wait(f"MOVE,{SAFE_POSE}")
        self.is_all_auto = False
        self.all_auto_btn.configure(text="🤖 ALL-AUTO START", fg_color="#e74c3c")

    def _execute_scoop_logic(self, bx, by):
        # 1. 현재 물체 위치에 맞는 최적의 RZ 각도 계산
        dynamic_rz = self.calculate_dynamic_rz(bx, by)
        
        # 2. 구역 판별
        if bx < 670 and by > 758: selected_area = AREA_A
        elif bx < 670 and by <= 758: selected_area = AREA_B
        else: selected_area = AREA_C
        
        scoop_vel = 250
        scoop_r = 50
        
        for step in selected_area:
            tx, ty, tz = bx + step['x'], by + step['y'], MOVE_Z_DEPTH + step['z']
            
            # [수정] 기존 step['rz'] 대신 계산된 dynamic_rz를 사용하거나, 
            # 기존 오프셋 rz를 dynamic_rz에 더해서 상대적 회전을 적용합니다.
            # 여기서는 계산된 dynamic_rz를 기본으로 하되 각 step의 미세 회전(offset)을 더합니다.
            final_rz = dynamic_rz + (step['rz'] - 90) # 90도는 기준점 보정용
            
            cmd = f"MOVE,{tx:.2f},{ty:.2f},{tz:.2f},{step['rx']:.2f},{step['ry']:.2f},{final_rz:.2f},{scoop_vel},{scoop_r}"
            
            print(f"DEBUG: Dynamic Scoop -> RZ: {final_rz:.2f} | CMD: {cmd}")
            if not self.send_command_and_wait(cmd): break

    def calculate_dynamic_j6(self, ball_x, ball_y):
        try:
            # 1. 원점 설정 (사용자 지정: 550, 350)
            center_pixel_x, center_pixel_y = 550, 350
            dx = ball_x - center_pixel_x
            dy = ball_y - center_pixel_y

            # 2. 이미지상 각도 계산
            angle_rad = math.atan2(-dy, dx)
            angle_deg = math.degrees(angle_rad)
            
            print(f"Current Raw Angle: {angle_deg:.2f}") 

            # 3. 목표 각도 계산 (정면 137.14 기준)
            base_j6 = 137.14
            offset = 180.0  # 아까 확인하신 오차값 적용
            target_j6 = base_j6 - (angle_deg - offset)

            # 4. [수정됨] 최단 경로 diff 계산 로직 추가
            # 이 줄이 빠져있어서 에러가 났었습니다!
            diff = (target_j6 - base_j6 + 180) % 360 - 180
            final_j6 = base_j6 + diff

            # 5. 안전 범위 제한
            limit = 350
            if final_j6 > limit: final_j6 = limit
            if final_j6 < -limit: final_j6 = -limit
            
            print(f"Final J6: {final_j6:.2f} (Diff from base: {diff:.2f})")
            return final_j6
            
        except Exception as e:
            print(f"Calculation Error: {e}")
            return 137.14 # 에러 발생 시 기본 정면 위치 반환
    def execute_rz_adjustment(self):
        """감지된 공 좌표를 J6 각도로 변환하여 실제로 로봇을 이동시킴"""
        # 1. 공 좌표 가져오기 (마지막으로 감지된 위치)
        pos = getattr(self, 'last_pixel_pos', None) or getattr(self, 'last_target_pos', None)
        
        if pos is None:
            self.status_label.configure(text="⚠️ No detection data!", text_color="#e74c3c")
            print("❌ 에러: 공이 감지되지 않아 이동할 수 없습니다.")
            return

        try:
            bx, by = pos[0], pos[1]
            
            # 2. [가장 중요] _move_worker를 호출하여 "ADJUST_RZ" 모드로 실행
            # 이 줄이 있어야 로봇에게 소켓 명령(MOVE,...)이 날아갑니다.
            threading.Thread(target=self._move_worker, 
                             args=("ADJUST_RZ", (bx, by)), 
                             daemon=True).start()
            
            print(f"🚀 로봇 이동 명령 전송 시작 (Target Pixel: {bx}, {by})")

        except Exception as e:
            print(f"❌ RZ 실행 에러: {e}")
            self.status_label.configure(text="⚠️ RZ Error", text_color="#e74c3c")
    def execute_scoop_motion(self):
        pos = getattr(self, 'last_pixel_pos', None) or getattr(self, 'last_target_pos', None)
        if pos is None:
            self.status_label.configure(text="⚠️ No detection data!", text_color="#e74c3c")
            return
        bx, by = pos[0], pos[1]
        threading.Thread(target=self._move_worker, args=("SCOOP_MOTION", (bx, by)), daemon=True).start()
    # 실행 함수들
    # [수정본] 중복 제거하고 하나로 통합된 버튼 함수들
    def execute_home_move(self): 
        threading.Thread(target=self._move_worker, args=("HOME", None), daemon=True).start()

    def execute_safe_move(self): 
        threading.Thread(target=self._move_worker, args=("SAFE", None), daemon=True).start()

    def execute_discharge_move(self): 
        threading.Thread(target=self._move_worker, args=("DISCHARGE", None), daemon=True).start()

    def execute_discharge_sequence(self): 
        threading.Thread(target=self._move_worker, args=("DISCHARGE_SEQ", None), daemon=True).start()

    def execute_screw_move(self): 
        threading.Thread(target=self._move_worker, args=("SCREW", None), daemon=True).start()

    def execute_sub_program_call(self): 
        threading.Thread(target=self._move_worker, args=("ddischarge", None), daemon=True).start()

    def execute_strike_move(self): 
        pos = list(self.last_target_pos) if self.last_target_pos else None
        threading.Thread(target=self._move_worker, args=("STRIKE", pos), daemon=True).start()

    def execute_dual_scoop(self): 
        pos = list(self.last_target_pos) if self.last_target_pos else None
        threading.Thread(target=self._move_worker, args=("DUAL_STEP", pos), daemon=True).start()
    def set_target_color(self, color_key): 
        self.target_color_key = color_key

    def execute_named_sub_program(self, cmd_name):
        threading.Thread(target=self._move_worker, args=(cmd_name, None), daemon=True).start()
    def execute_marker_move(self):
        """냄비 센터 고정 위치로 이동"""
        # x, y, z, rot 정보를 리스트로 묶어서 하나의 인자(pos)로 전달합니다.
        # _move_worker(self, mode, target_pos_at_click) 구조에 맞춤
        fixed_pos = [812.0, 951.0, 828.42, "74.74,167.13,-13.04"]
        
        self.status_label.configure(text="🚀 Moving to POT CENTER", text_color="#27ae60")
        # 인자를 ("FIXED_POS", fixed_pos) 딱 두 개만 던집니다.
        threading.Thread(target=self._move_worker, args=("FIXED_POS", fixed_pos), daemon=True).start()
    def calculate_polar_j6(self, ball_x, ball_y):
        try:
            center_x, center_y = 550, 350
            dx = ball_x - center_x
            dy = ball_y - center_y
            angle_deg = math.degrees(math.atan2(dy, dx))

            # [0]~[8] 구간 (각도 작은 쪽)
            MAP_0_8 = [
                (1.2,   -12.26),   # [0]
                (18.8,  -52.66),   # [2]
                (29.0,  -64.95),   # [3]
                (49.1,  -80.99),   # [6]
                (57.1,  -152.99),
            ]

            # [9]~[17] 구간 (각도 큰 쪽)
            MAP_9_17 = [
                (80.9, 161.00),
                (92.9, 137.00),
                (140.1, 40.90),   # [14]
                (154.1, 2),  # [17]
            ]

            # 구간 판별 기준각도 (실측 필요)
            BOUNDARY = 67.3  # [8]과 [9] 사이

            if angle_deg <= BOUNDARY:
                table = MAP_0_8
            else:
                table = MAP_9_17

            # 선형 보간
            if angle_deg <= table[0][0]:
                final_j6 = table[0][1]
            elif angle_deg >= table[-1][0]:
                final_j6 = table[-1][1]
            else:
                for i in range(len(table) - 1):
                    a1, j1 = table[i]
                    a2, j2 = table[i+1]
                    if a1 <= angle_deg <= a2:
                        t = (angle_deg - a1) / (a2 - a1)
                        final_j6 = j1 + t * (j2 - j1)
                        break

            print(f"🎯 Raw Angle: {angle_deg:.1f} | J6: {final_j6:.2f}")
            return final_j6

        except Exception as e:
            print(f"❌ 계산 오류: {e}")
            return -12.26
if __name__ == "__main__":
    ctk.set_appearance_mode("Dark")
    RobotApp().mainloop()