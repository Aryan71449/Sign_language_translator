import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import customtkinter as ctk
from PIL import Image
import os
import time
import warnings

warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
DATA_PATH = "sign_data.csv"
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

FONT_HEADER = ("Poppins", 24, "bold")
FONT_SUBHEADER = ("Poppins", 16, "bold")
FONT_BODY = ("Poppins", 12)
FONT_BUTTON = ("Poppins", 13, "bold")

def normalize_landmarks(landmarks):
    pts = np.array([(lm.x, lm.y) for lm in landmarks], dtype=np.float32)
    pts -= pts[0]
    scale = np.max(np.linalg.norm(pts, axis=1)) + 1e-6
    pts /= scale
    return pts.flatten()

class PolishedDataCollector(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Neuro Talk Collector")
        self.geometry("1100x700")

        self.cap = None
        self.running = False
        self.collecting = False
        self.samples = 0
        self.last_save = 0
        self.hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                                    min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.init_ui()

    def init_ui(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.sidebar = ctk.CTkFrame(self, width=280, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_rowconfigure(5, weight=1)

        ctk.CTkLabel(self.sidebar, text="Neuro Talk", font=FONT_HEADER).grid(row=0, column=0, padx=20, pady=(30, 20))
        ctk.CTkLabel(self.sidebar, text="Label Name (e.g. 'A')", font=FONT_BODY, anchor="w").grid(row=1, column=0, padx=20, pady=(10, 0), sticky="w")
        self.label_entry = ctk.CTkEntry(self.sidebar, placeholder_text="Type label here...", font=FONT_BODY, height=40)
        self.label_entry.grid(row=2, column=0, padx=20, pady=(5, 20), sticky="ew")

        self.stats_card = ctk.CTkFrame(self.sidebar, fg_color=("gray85", "gray25"))
        self.stats_card.grid(row=3, column=0, padx=20, pady=10, sticky="ew")
        ctk.CTkLabel(self.stats_card, text="Samples Collected", font=("Poppins", 11)).pack(pady=(10, 0))
        self.count_label = ctk.CTkLabel(self.stats_card, text="0", font=("Poppins", 32, "bold"), text_color="#3B8ED0")
        self.count_label.pack(pady=(0, 10))

        self.appearance_mode_optionemenu = ctk.CTkOptionMenu(self.sidebar, values=["Dark", "Light"], 
        command=self.change_appearance_mode_event, font=FONT_BODY)
        self.appearance_mode_optionemenu.grid(row=6, column=0, padx=20, pady=30)

        self.main_frame = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.main_frame.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
        self.main_frame.grid_rowconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)

        # --- VIDEO CARD (Fill Mode) ---
        self.video_card = ctk.CTkFrame(self.main_frame, corner_radius=20, fg_color="black")
        self.video_card.grid(row=0, column=0, sticky="nsew")
        
        # PADDING REMOVED
        self.video_label = ctk.CTkLabel(self.video_card, text="", corner_radius=20)
        self.video_label.pack(fill="both", expand=True, padx=0, pady=0)

        self.ctrl_bar = ctk.CTkFrame(self.main_frame, height=80, corner_radius=15)
        self.ctrl_bar.grid(row=1, column=0, sticky="ew", pady=(20, 0))
        
        self.btn_cam = ctk.CTkButton(self.ctrl_bar, text="Start Camera", command=self.toggle_camera, 
                                     font=FONT_BUTTON, height=45, corner_radius=10)
        self.btn_cam.pack(side="left", padx=20, pady=15, expand=True, fill="x")

        self.btn_rec = ctk.CTkButton(self.ctrl_bar, text="Start Recording", command=self.toggle_collect, 
                                     state="disabled", font=FONT_BUTTON, height=45, corner_radius=10, 
                                     fg_color="#2CC985", hover_color="#229964")
        self.btn_rec.pack(side="left", padx=20, pady=15, expand=True, fill="x")

    def change_appearance_mode_event(self, new_appearance_mode: str):
        ctk.set_appearance_mode(new_appearance_mode)

    def toggle_camera(self):
        if self.running:
            self.running = False
            self.collecting = False
            if self.cap: self.cap.release(); self.cap = None
            self.video_label.configure(image=None)
            self.btn_cam.configure(text="Start Camera", fg_color=["#3B8ED0", "#1F6AA5"])
            self.btn_rec.configure(state="disabled", text="Start Recording", fg_color="#2CC985")
            self.label_entry.configure(state="normal")
        else:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened(): return
            self.running = True
            self.btn_cam.configure(text="Stop Camera", fg_color="#C0392B", hover_color="#E74C3C")
            self.btn_rec.configure(state="normal")
            self.update_frame()

    def toggle_collect(self):
        if not self.collecting:
            label = self.label_entry.get().strip()
            if not label:
                self.label_entry.configure(placeholder_text_color="red")
                return
            self.samples = 0
            self.collecting = True
            self.btn_rec.configure(text="⏹ Stop Recording", fg_color="#C0392B", hover_color="#E74C3C")
            self.label_entry.configure(state="disabled")
        else:
            self.collecting = False
            self.btn_rec.configure(text="⏺ Start Recording", fg_color="#2CC985", hover_color="#229964")
            self.label_entry.configure(state="normal")

    def update_frame(self):
        if not self.running: return
        ret, frame = self.cap.read()
        if not ret: self.toggle_camera(); return

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.hands.process(rgb)

        if res.multi_hand_landmarks:
            for hand in res.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
                if self.collecting and (time.time() - self.last_save > 0.05):
                    feat = normalize_landmarks(hand.landmark)
                    row = np.append(feat, self.label_entry.get().strip())
                    if not os.path.exists(DATA_PATH):
                        pd.DataFrame([row]).to_csv(DATA_PATH, mode='w', header=False, index=False)
                    else:
                        pd.DataFrame([row]).to_csv(DATA_PATH, mode='a', header=False, index=False)
                    self.samples += 1
                    self.count_label.configure(text=str(self.samples))
                    self.last_save = time.time()

        if self.collecting:
            cv2.circle(frame, (30, 30), 10, (255, 0, 0), -1) 
            cv2.putText(frame, "REC", (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        # --- FIXED SIZE ---
        ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(800, 600))
        self.video_label.configure(image=ctk_img)
        self.video_label.image = ctk_img
        
        self.after(15, self.update_frame)

if __name__ == "__main__":
    app = PolishedDataCollector()
    app.mainloop()

