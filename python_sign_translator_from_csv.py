import os
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from sklearn.ensemble import RandomForestClassifier
import joblib
import customtkinter as ctk
from PIL import Image
import threading
import pygame
from gtts import gTTS
from deep_translator import GoogleTranslator
from collections import deque, Counter
import warnings

warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
DATA_PATH = "sign_data.csv"
MODEL_PATH = "sign_model.pkl"
VIDEO_FOLDER = "video"
if not os.path.exists(VIDEO_FOLDER): os.makedirs(VIDEO_FOLDER)

# Mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# --- UI CONSTANTS ---
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

class PolishedTranslatorApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Neuro Talk")
        self.geometry("1250x850")
        
        pygame.mixer.init()

        self.cap = None
        self.running = False
        self.hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                                    min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.clf = None
        self.window = deque(maxlen=12)
        self.last_prediction = None
        self.video_cap = None
        
        self.init_ui()
        threading.Thread(target=self.prepare_model, daemon=True).start()

    def init_ui(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.sidebar = ctk.CTkFrame(self, width=250, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_rowconfigure(4, weight=1)

        self.logo_label = ctk.CTkLabel(self.sidebar, text="Neuro Talk", font=FONT_HEADER)
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        self.btn_translate = ctk.CTkButton(self.sidebar, text="Translate Mode", command=self.show_translate_view, font=FONT_BUTTON, corner_radius=8, height=40)
        self.btn_translate.grid(row=1, column=0, padx=20, pady=10, sticky="ew")

        self.btn_learn = ctk.CTkButton(self.sidebar, text="Learning Mode", command=self.show_learning_view, font=FONT_BUTTON, corner_radius=8, height=40, fg_color="transparent", border_width=2, text_color=("gray10", "gray90"))
        self.btn_learn.grid(row=2, column=0, padx=20, pady=10, sticky="ew")

        self.appearance_mode_label = ctk.CTkLabel(self.sidebar, text="Appearance Mode:", anchor="w", font=FONT_BODY)
        self.appearance_mode_label.grid(row=5, column=0, padx=20, pady=(10, 0))
        self.appearance_mode_optionemenu = ctk.CTkOptionMenu(self.sidebar, values=["Dark", "Light"], command=self.change_appearance_mode_event, font=FONT_BODY)
        self.appearance_mode_optionemenu.grid(row=6, column=0, padx=20, pady=(10, 20))

        self.main_frame = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.main_frame.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
        
        self.translate_view = self.create_translate_view()
        self.learning_view = self.create_learning_view()
        self.show_translate_view()

    def create_translate_view(self):
        frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        frame.grid_columnconfigure(0, weight=3)
        frame.grid_columnconfigure(1, weight=1)
        frame.grid_rowconfigure(0, weight=1)

        # --- VIDEO CARD (Fill Mode) ---
        # corner_radius=20 gives the rounded look.
        # clips_children=True isn't supported in CTk natively yet, 
        # so we rely on the image filling the frame cleanly.
        self.vid_card = ctk.CTkFrame(frame, corner_radius=20, fg_color="black") 
        self.vid_card.grid(row=0, column=0, sticky="nsew", padx=(0, 20))
        
        # REMOVED PADDING HERE: padx=0, pady=0 allows the image to touch the edges
        self.video_label = ctk.CTkLabel(self.vid_card, text="", corner_radius=20)
        self.video_label.pack(fill="both", expand=True, padx=0, pady=0)

        # --- CONTROLS ---
        self.ctrl_panel = ctk.CTkFrame(frame, corner_radius=15)
        self.ctrl_panel.grid(row=0, column=1, sticky="nsew")
        
        ctk.CTkLabel(self.ctrl_panel, text="Detected Sign", font=FONT_SUBHEADER, text_color="gray").pack(pady=(20, 5), padx=20, anchor="w")
        self.pred_label = ctk.CTkLabel(self.ctrl_panel, text="...", font=("Poppins", 40, "bold"), text_color="#3B8ED0")
        self.pred_label.pack(pady=5)
        self.status_label = ctk.CTkLabel(self.ctrl_panel, text="Loading Model...", font=("Poppins", 10))
        self.status_label.pack(pady=(0, 20))

        ctk.CTkLabel(self.ctrl_panel, text="Transcript", font=FONT_SUBHEADER, text_color="gray").pack(pady=(10, 5), padx=20, anchor="w")
        self.transcript = ctk.CTkTextbox(self.ctrl_panel, font=FONT_BODY, corner_radius=10, height=150)
        self.transcript.pack(fill="x", padx=20, pady=5)

        self.btn_cam = ctk.CTkButton(self.ctrl_panel, text="Start Camera", command=self.toggle_camera, font=FONT_BUTTON, height=45, corner_radius=10)
        self.btn_cam.pack(fill="x", padx=20, pady=(20, 10))

        speak_frame = ctk.CTkFrame(self.ctrl_panel, fg_color="transparent")
        speak_frame.pack(fill="x", padx=20, pady=5)
        ctk.CTkButton(speak_frame, text="🔊 Eng", command=lambda: self.speak("en"), width=60, font=FONT_BUTTON).pack(side="left", fill="x", expand=True, padx=(0, 5))
        ctk.CTkButton(speak_frame, text="🔊 Hindi", command=lambda: self.speak("hi"), width=60, font=FONT_BUTTON, fg_color="#E07A5F", hover_color="#C0583E").pack(side="left", fill="x", expand=True, padx=(5, 0))

        ctk.CTkButton(self.ctrl_panel, text="Clear Text", command=self.reset_transcript, font=FONT_BUTTON, fg_color="transparent", border_width=1).pack(fill="x", padx=20, pady=10)

        return frame

    def create_learning_view(self):
        frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_columnconfigure(1, weight=3)
        frame.grid_rowconfigure(0, weight=1)

        list_card = ctk.CTkFrame(frame, corner_radius=15)
        list_card.grid(row=0, column=0, sticky="nsew", padx=(0, 20))
        ctk.CTkLabel(list_card, text="Video Library", font=FONT_SUBHEADER).pack(pady=20, padx=20, anchor="w")
        self.scroll_list = ctk.CTkScrollableFrame(list_card, label_text="Lessons")
        self.scroll_list.pack(fill="both", expand=True, padx=10, pady=10)
        self.refresh_video_list()

        self.player_card = ctk.CTkFrame(frame, corner_radius=15, fg_color="black")
        self.player_card.grid(row=0, column=1, sticky="nsew")
        
        # PADDING REMOVED HERE TOO
        self.learn_video_label = ctk.CTkLabel(self.player_card, text="Select a video to play", corner_radius=15)
        self.learn_video_label.pack(fill="both", expand=True, padx=0, pady=0)
        
        ctrls = ctk.CTkFrame(frame, fg_color="transparent")
        ctrls.grid(row=1, column=1, pady=10) # Moved controls below video to keep video pure
        ctk.CTkButton(ctrls, text="▶ Play", command=self.play_video, width=100).pack(side="left", padx=10)
        ctk.CTkButton(ctrls, text="⏹ Stop", command=self.stop_video, width=100, fg_color="#C0392B", hover_color="#E74C3C").pack(side="left", padx=10)

        return frame

    def show_translate_view(self):
        self.stop_video()
        self.learning_view.pack_forget()
        self.translate_view.pack(fill="both", expand=True)
        self.btn_translate.configure(fg_color=["#3B8ED0", "#1F6AA5"])
        self.btn_learn.configure(fg_color="transparent")

    def show_learning_view(self):
        self.stop_camera_logic()
        self.translate_view.pack_forget()
        self.learning_view.pack(fill="both", expand=True)
        self.btn_translate.configure(fg_color="transparent")
        self.btn_learn.configure(fg_color=["#3B8ED0", "#1F6AA5"])
        self.refresh_video_list()

    def change_appearance_mode_event(self, new_appearance_mode: str):
        ctk.set_appearance_mode(new_appearance_mode)

    def prepare_model(self):
        if os.path.exists(MODEL_PATH):
            try:
                self.clf = joblib.load(MODEL_PATH)
                if hasattr(self.clf, "classes_"): self.labels_ = self.clf.classes_
                self.after(0, lambda: self.status_label.configure(text="AI Ready", text_color="#2CC985"))
                return
            except: pass
        self.after(0, lambda: self.status_label.configure(text="Training AI...", text_color="orange"))
        if os.path.exists(DATA_PATH):
            try:
                df = pd.read_csv(DATA_PATH, header=None)
                X, y = df.iloc[:, :-1].values, df.iloc[:, -1].values
                self.clf = RandomForestClassifier(n_estimators=100, random_state=42)
                self.clf.fit(X, y)
                self.labels_ = np.unique(y)
                joblib.dump(self.clf, MODEL_PATH)
                self.after(0, lambda: self.status_label.configure(text="AI Ready", text_color="#2CC985"))
            except: self.after(0, lambda: self.status_label.configure(text="Training Failed", text_color="red"))

    def toggle_camera(self):
        if self.running:
            self.stop_camera_logic()
            self.btn_cam.configure(text="Start Camera", fg_color=["#3B8ED0", "#1F6AA5"])
        else:
            if self.clf is None: return
            self.cap = cv2.VideoCapture(0)
            self.running = True
            self.btn_cam.configure(text="Stop Camera", fg_color="#C0392B", hover_color="#E74C3C")
            self.update_translate()

    def stop_camera_logic(self):
        self.running = False
        if self.cap: self.cap.release(); self.cap = None
        self.video_label.configure(image=None)
        self.window.clear()

    def update_translate(self):
        if not self.running: return
        ok, frame = self.cap.read()
        if not ok: self.toggle_camera(); return

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.hands.process(rgb)
        
        pred_text = "..."
        if res.multi_hand_landmarks:
            for hand in res.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
                try:
                    feat = normalize_landmarks(hand.landmark)
                    proba = self.clf.predict_proba([feat])[0]
                    idx = int(np.argmax(proba))
                    if proba[idx] > 0.6: self.window.append(self.clf.classes_[idx])
                except: pass

            if len(self.window) == self.window.maxlen:
                common, count = Counter(self.window).most_common(1)[0]
                if count > 8: pred_text = common

        if pred_text != "..." and pred_text != self.last_prediction:
            self.pred_label.configure(text=pred_text)
            self.transcript.insert("end", pred_text + " ")
            self.transcript.see("end")
            self.last_prediction = pred_text
        elif pred_text == "...":
            self.pred_label.configure(text="...")

        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        # --- FIXED SIZE FOR "FILL MODE" ---
        # 800x600 is arbitrary but large enough to look like it fills the space
        ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(800, 600))
        self.video_label.configure(image=ctk_img)
        self.video_label.image = ctk_img 
        
        self.after(15, self.update_translate)

    def reset_transcript(self):
        self.transcript.delete("0.0", "end")
        self.last_prediction = None

    def speak(self, lang):
        txt = self.transcript.get("0.0", "end").strip()
        if not txt: return
        threading.Thread(target=self._threaded_speak, args=(txt, lang), daemon=True).start()

    def _threaded_speak(self, text, lang):
        try:
            target_txt, l, tld = text, 'en', 'co.in'
            if lang == 'hi':
                target_txt = GoogleTranslator(source='auto', target='hi').translate(text)
                l, tld = 'hi', 'com'
            tts = gTTS(text=target_txt, lang=l, tld=tld)
            tts.save("voice.mp3")
            pygame.mixer.music.load("voice.mp3")
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy(): pygame.time.Clock().tick(10)
            pygame.mixer.music.unload()
            os.remove("voice.mp3")
        except: pass

    def refresh_video_list(self):
        for widget in self.scroll_list.winfo_children(): widget.destroy()
        files = [f for f in os.listdir(VIDEO_FOLDER) if f.endswith(".mp4")]
        if not files:
            ctk.CTkLabel(self.scroll_list, text="No videos found").pack(pady=10)
            return
        for f in files:
            ctk.CTkButton(self.scroll_list, text=f, command=lambda fname=f: self.select_video(fname),
                          fg_color="transparent", border_width=1, text_color=("black", "white"), anchor="w").pack(fill="x", pady=2)

    def select_video(self, fname):
        self.current_video_file = fname
        self.stop_video()
        self.play_video()

    def play_video(self):
        if not hasattr(self, 'current_video_file'): return
        path = os.path.join(VIDEO_FOLDER, self.current_video_file)
        self.video_cap = cv2.VideoCapture(path)
        self.update_video_frame()

    def stop_video(self):
        if self.video_cap: self.video_cap.release(); self.video_cap = None
    
    def update_video_frame(self):
        if not self.video_cap or not self.video_cap.isOpened(): return
        ret, frame = self.video_cap.read()
        if not ret: self.stop_video(); return
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        # FIXED SIZE HERE TOO
        ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(800, 600))
        self.learn_video_label.configure(image=ctk_img, text="")
        self.learn_video_label.image = ctk_img
        self.after(30, self.update_video_frame)

if __name__ == "__main__":
    app = PolishedTranslatorApp()
    app.mainloop()

