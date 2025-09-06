# # import os
# # import cv2
# # import numpy as np
# # import pandas as pd
# # import mediapipe as mp
# # from sklearn.ensemble import RandomForestClassifier
# # from sklearn.model_selection import train_test_split
# # from sklearn.metrics import accuracy_score
# # import joblib
# # import tkinter as tk
# # from tkinter import ttk, messagebox, filedialog
# # from PIL import Image, ImageTk
# # import pyttsx3
# # from collections import deque, Counter

# # DATA_PATH = "sign_data.csv"
# # MODEL_PATH = "sign_model.pkl"

# # mp_hands = mp.solutions.hands
# # mp_drawing = mp.solutions.drawing_utils

# # # Light UI Colors
# # BG_COLOR = "#ffffff"        # window background
# # BTN_COLOR = "#3b82f6"       # button blue
# # TEXT_COLOR = "#000000"      # black text
# # ACCENT_COLOR = "#16a34a"    # green accent

# # def normalize_landmarks(landmarks):
# #     pts = np.array([(lm.x, lm.y) for lm in landmarks], dtype=np.float32)
# #     origin = pts[0]
# #     pts -= origin
# #     scale = np.max(np.linalg.norm(pts, axis=1)) + 1e-6
# #     pts /= scale
# #     return pts.flatten()

# # class SignTranslatorFromCSV:
# #     def __init__(self, root):
# #         self.root = root
# #         self.root.title("🖐 Sign Language Translator")
# #         self.root.geometry("1050x800")
# #         self.root.configure(bg=BG_COLOR)

# #         style = ttk.Style()
# #         style.theme_use("clam")
# #         style.configure("TButton", padding=10, relief="flat", background=BTN_COLOR, foreground=TEXT_COLOR)
# #         style.configure("Big.TButton",
# #                         padding=15,
# #                         relief="flat",
# #                         background=BTN_COLOR,
# #                         foreground=TEXT_COLOR,
# #                         font=("Segoe UI", 12, "bold"))
# #         style.map("Big.TButton", background=[("active", "#2563eb")])

# #         self.video_label = tk.Label(self.root, bg="black")
# #         self.video_label.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

# #         controls = ttk.Frame(self.root)
# #         controls.pack(fill=tk.X, pady=10)

# #         self.start_btn = ttk.Button(controls, text="▶ Start", command=self.start, style="Big.TButton")
# #         self.stop_btn = ttk.Button(controls, text="⏹ Stop", command=self.stop, style="Big.TButton")
# #         self.speak_btn = ttk.Button(controls, text="🔊 Speak", command=self.speak, style="Big.TButton")
# #         self.reset_btn = ttk.Button(controls, text="♻ Reset Transcript", command=self.reset_transcript, style="Big.TButton")
# #         self.save_btn = ttk.Button(controls, text="💾 Save Transcript", command=self.save_transcript, style="Big.TButton")

# #         for btn in [self.start_btn, self.stop_btn, self.speak_btn, self.reset_btn, self.save_btn]:
# #             btn.pack(side=tk.LEFT, padx=3, pady=3, fill=tk.X, expand=True)

# #         status_frame = ttk.Frame(self.root)
# #         status_frame.pack(fill=tk.X, pady=5)
# #         ttk.Label(status_frame, text="Prediction:", font=("Segoe UI", 14, "bold"),
# #                   foreground=TEXT_COLOR, background=BG_COLOR).pack(side=tk.LEFT, padx=5)
# #         self.pred_var = tk.StringVar(value="—")
# #         self.pred_label = ttk.Label(status_frame, textvariable=self.pred_var,
# #                                     font=("Segoe UI", 18, "bold"), foreground=ACCENT_COLOR, background=BG_COLOR)
# #         self.pred_label.pack(side=tk.LEFT, padx=10)
# #         self.conf_var = tk.StringVar(value="")
# #         ttk.Label(status_frame, textvariable=self.conf_var, font=("Segoe UI", 12),
# #                   foreground="#4b5563", background=BG_COLOR).pack(side=tk.LEFT)

# #         transcript_frame = ttk.LabelFrame(self.root, text="Transcript")
# #         transcript_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
# #         self.transcript = tk.Text(transcript_frame, height=8, font=("Segoe UI", 12),
# #                                   bg="#f3f4f6", fg=TEXT_COLOR, insertbackground="black")
# #         self.transcript.pack(fill=tk.BOTH, expand=True)

# #         self.cap = None
# #         self.running = False
# #         self.hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
# #                                      min_detection_confidence=0.5, min_tracking_confidence=0.5)
# #         self.engine = pyttsx3.init()
# #         self.engine.setProperty("rate", 170)

# #         self.clf = None
# #         self.labels_ = None
# #         self.window = deque(maxlen=12)
# #         self.last_prediction = None

# #         self.prepare_and_train()

# #     def prepare_and_train(self):
# #         if not os.path.exists(DATA_PATH):
# #             messagebox.showerror("Dataset Missing", f"{DATA_PATH} not found.")
# #             return
# #         df = pd.read_csv(DATA_PATH, header=None)
# #         if df.empty:
# #             messagebox.showerror("Dataset Empty", f"{DATA_PATH} is empty.")
# #             return
# #         X = df.iloc[:, :-1].values
# #         y = df.iloc[:, -1].values
# #         try:
# #             Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
# #         except Exception:
# #             Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
# #         self.clf = RandomForestClassifier(n_estimators=300, random_state=42)
# #         self.clf.fit(Xtr, ytr)
# #         if len(np.unique(yte)) > 1 and len(yte) > 0:
# #             yhat = self.clf.predict(Xte)
# #             acc = accuracy_score(yte, yhat)
# #             self.conf_var.set(f"Model Accuracy: {acc:.2f}")
# #         self.labels_ = np.unique(y)
# #         joblib.dump(self.clf, MODEL_PATH)

# #     def start(self):
# #         if self.running:
# #             return
# #         if self.clf is None:
# #             messagebox.showerror("No Model", "Train the model first.")
# #             return
# #         self.cap = cv2.VideoCapture(0)
# #         if not self.cap.isOpened():
# #             messagebox.showerror("Camera Error", "Could not open webcam.")
# #             return
# #         self.running = True
# #         self.update()

# #     def stop(self):
# #         self.running = False
# #         if self.cap:
# #             self.cap.release()
# #         self.video_label.config(image='')
# #         self.window.clear()

# #     def reset_transcript(self):
# #         self.transcript.delete("1.0", tk.END)
# #         self.last_prediction = None

# #     def update(self):
# #         if not self.running:
# #             return
# #         ok, frame = self.cap.read()
# #         if not ok:
# #             self.stop()
# #             return
# #         frame = cv2.flip(frame, 1)
# #         rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
# #         res = self.hands.process(rgb)
# #         pred = None
# #         pred_conf = 0.0
# #         if res.multi_hand_landmarks:
# #             hand = res.multi_hand_landmarks[0]
# #             mp_drawing.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
# #             feat = normalize_landmarks(hand.landmark)
# #             if hasattr(self.clf, "predict_proba"):
# #                 proba = self.clf.predict_proba([feat])[0]
# #                 pred_idx = int(np.argmax(proba))
# #                 pred = self.clf.classes_[pred_idx]
# #                 pred_conf = float(proba[pred_idx])
# #             else:
# #                 pred = self.clf.predict([feat])[0]
# #                 pred_conf = 1.0
# #             self.window.append(pred)
# #             if len(self.window) == self.window.maxlen:
# #                 common, count = Counter(self.window).most_common(1)[0]
# #                 pred = common
# #                 pred_conf = count / len(self.window)
# #         if pred is not None:
# #             self.pred_var.set(str(pred))
# #             if pred_conf >= 0.7 and pred != self.last_prediction:
# #                 self.transcript.insert(tk.END, pred + ' ')
# #                 self.transcript.see(tk.END)
# #                 self.last_prediction = pred
# #         else:
# #             self.pred_var.set("—")
# #             self.last_prediction = None
# #         img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
# #         imgtk = ImageTk.PhotoImage(image=img)
# #         self.video_label.imgtk = imgtk
# #         self.video_label.configure(image=imgtk)
# #         self.root.after(15, self.update)

# #     def save_transcript(self):
# #         txt = self.transcript.get("1.0", tk.END).strip()
# #         if not txt:
# #             return
# #         path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text Files", "*.txt")])
# #         if path:
# #             with open(path, "w", encoding="utf-8") as f:
# #                 f.write(txt)
# #             messagebox.showinfo("Saved", f"Transcript saved to:\n{path}")

# #     def speak(self):
# #         txt = self.transcript.get("1.0", tk.END).strip()
# #         if not txt:
# #             return
# #         self.engine.say(txt)
# #         self.engine.runAndWait()

# #     def on_closing(self):
# #         if messagebox.askokcancel("Exit", "Are you sure you want to exit?"):
# #             self.stop()
# #             self.root.destroy()

# # if __name__ == "__main__":
# #     root = tk.Tk()
# #     app = SignTranslatorFromCSV(root)
# #     root.protocol("WM_DELETE_WINDOW", app.on_closing)
# #     root.mainloop()


# import os
# import cv2
# import numpy as np
# import pandas as pd
# import mediapipe as mp
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# import joblib
# import tkinter as tk
# from tkinter import ttk, messagebox, filedialog
# from PIL import Image, ImageTk
# import pyttsx3
# from collections import deque, Counter

# # Paths
# DATA_PATH = "sign_data.csv"
# MODEL_PATH = "sign_model.pkl"
# VIDEO_FOLDER = "video"

# # Ensure video folder exists
# if not os.path.exists(VIDEO_FOLDER):
#     os.makedirs(VIDEO_FOLDER)

# # Mediapipe
# mp_hands = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils

# # UI Colors
# BG_COLOR = "#ffffff"
# BTN_COLOR = "#3b82f6"
# TEXT_COLOR = "#000000"
# ACCENT_COLOR = "#16a34a"


# def normalize_landmarks(landmarks):
#     pts = np.array([(lm.x, lm.y) for lm in landmarks], dtype=np.float32)
#     origin = pts[0]
#     pts -= origin
#     scale = np.max(np.linalg.norm(pts, axis=1)) + 1e-6
#     pts /= scale
#     return pts.flatten()


# class SignTranslatorApp:
#     def __init__(self, root):
#         self.root = root
#         self.root.title("🖐 Sign Language Translator")
#         self.root.geometry("1050x800")
#         self.root.configure(bg=BG_COLOR)

#         # common vars
#         self.cap = None
#         self.running = False
#         self.hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
#                                     min_detection_confidence=0.5, min_tracking_confidence=0.5)
#         self.engine = pyttsx3.init()
#         self.engine.setProperty("rate", 170)
#         self.clf = None
#         self.labels_ = None
#         self.window = deque(maxlen=12)
#         self.last_prediction = None
#         self.video_cap = None

#         self.show_main_menu()

#     # ---------------- Main Menu ----------------
#     def show_main_menu(self):
#         self.clear_root()

#         frame = ttk.Frame(self.root)
#         frame.pack(expand=True)

#         ttk.Label(frame, text="🖐 Sign Language Translator", font=("Segoe UI", 20, "bold")).pack(pady=20)

#         ttk.Button(frame, text="Translate Mode", command=self.init_translate_ui,
#                    style="TButton").pack(pady=10, ipadx=20, ipady=10)
#         ttk.Button(frame, text="Learning Mode", command=self.init_learning_ui,
#                    style="TButton").pack(pady=10, ipadx=20, ipady=10)
#         ttk.Button(frame, text="Exit", command=self.root.quit,
#                    style="TButton").pack(pady=10, ipadx=20, ipady=10)

#     def clear_root(self):
#         for widget in self.root.winfo_children():
#             widget.destroy()

#     # ---------------- Translate Mode ----------------
#     def init_translate_ui(self):
#         self.clear_root()

#         self.video_label = tk.Label(self.root, bg="black")
#         self.video_label.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

#         controls = ttk.Frame(self.root)
#         controls.pack(fill=tk.X, pady=10)

#         self.start_btn = ttk.Button(controls, text="▶ Start", command=self.start)
#         self.stop_btn = ttk.Button(controls, text="⏹ Stop", command=self.stop)
#         self.speak_btn = ttk.Button(controls, text="🔊 Speak", command=self.speak)
#         self.reset_btn = ttk.Button(controls, text="♻ Reset Transcript", command=self.reset_transcript)
#         self.save_btn = ttk.Button(controls, text="💾 Save Transcript", command=self.save_transcript)
#         self.back_btn = ttk.Button(controls, text="⬅ Back", command=self.show_main_menu)

#         for btn in [self.start_btn, self.stop_btn, self.speak_btn,
#                     self.reset_btn, self.save_btn, self.back_btn]:
#             btn.pack(side=tk.LEFT, padx=3, pady=3, fill=tk.X, expand=True)

#         status_frame = ttk.Frame(self.root)
#         status_frame.pack(fill=tk.X, pady=5)
#         ttk.Label(status_frame, text="Prediction:", font=("Segoe UI", 14, "bold")).pack(side=tk.LEFT, padx=5)
#         self.pred_var = tk.StringVar(value="—")
#         self.pred_label = ttk.Label(status_frame, textvariable=self.pred_var,
#                                     font=("Segoe UI", 18, "bold"), foreground=ACCENT_COLOR)
#         self.pred_label.pack(side=tk.LEFT, padx=10)
#         self.conf_var = tk.StringVar(value="")
#         ttk.Label(status_frame, textvariable=self.conf_var, font=("Segoe UI", 12),
#                   foreground="#4b5563").pack(side=tk.LEFT)

#         transcript_frame = ttk.LabelFrame(self.root, text="Transcript")
#         transcript_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
#         self.transcript = tk.Text(transcript_frame, height=8, font=("Segoe UI", 12),
#                                   bg="#f3f4f6", fg=TEXT_COLOR, insertbackground="black")
#         self.transcript.pack(fill=tk.BOTH, expand=True)

#         self.prepare_and_train()

#     def prepare_and_train(self):
#         if not os.path.exists(DATA_PATH):
#             messagebox.showerror("Dataset Missing", f"{DATA_PATH} not found.")
#             return
#         df = pd.read_csv(DATA_PATH, header=None)
#         if df.empty:
#             messagebox.showerror("Dataset Empty", f"{DATA_PATH} is empty.")
#             return
#         X = df.iloc[:, :-1].values
#         y = df.iloc[:, -1].values
#         try:
#             Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
#         except Exception:
#             Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
#         self.clf = RandomForestClassifier(n_estimators=300, random_state=42)
#         self.clf.fit(Xtr, ytr)
#         if len(np.unique(yte)) > 1 and len(yte) > 0:
#             yhat = self.clf.predict(Xte)
#             acc = accuracy_score(yte, yhat)
#             self.conf_var.set(f"Model Accuracy: {acc:.2f}")
#         self.labels_ = np.unique(y)
#         joblib.dump(self.clf, MODEL_PATH)

#     def start(self):
#         if self.running:
#             return
#         if self.clf is None:
#             messagebox.showerror("No Model", "Train the model first.")
#             return
#         self.cap = cv2.VideoCapture(0)
#         if not self.cap.isOpened():
#             messagebox.showerror("Camera Error", "Could not open webcam.")
#             return
#         self.running = True
#         self.update_translate()

#     def stop(self):
#         self.running = False
#         if self.cap:
#             self.cap.release()
#         self.video_label.config(image='')
#         self.window.clear()

#     def reset_transcript(self):
#         self.transcript.delete("1.0", tk.END)
#         self.last_prediction = None

#     def update_translate(self):
#         if not self.running:
#             return
#         ok, frame = self.cap.read()
#         if not ok:
#             self.stop()
#             return
#         frame = cv2.flip(frame, 1)
#         rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         res = self.hands.process(rgb)
#         pred = None
#         pred_conf = 0.0
#         if res.multi_hand_landmarks:
#             hand = res.multi_hand_landmarks[0]
#             mp_drawing.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
#             feat = normalize_landmarks(hand.landmark)
#             if hasattr(self.clf, "predict_proba"):
#                 proba = self.clf.predict_proba([feat])[0]
#                 pred_idx = int(np.argmax(proba))
#                 pred = self.clf.classes_[pred_idx]
#                 pred_conf = float(proba[pred_idx])
#             else:
#                 pred = self.clf.predict([feat])[0]
#                 pred_conf = 1.0
#             self.window.append(pred)
#             if len(self.window) == self.window.maxlen:
#                 common, count = Counter(self.window).most_common(1)[0]
#                 pred = common
#                 pred_conf = count / len(self.window)
#         if pred is not None:
#             self.pred_var.set(str(pred))
#             if pred_conf >= 0.7 and pred != self.last_prediction:
#                 self.transcript.insert(tk.END, pred + ' ')
#                 self.transcript.see(tk.END)
#                 self.last_prediction = pred
#         else:
#             self.pred_var.set("—")
#             self.last_prediction = None
#         img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#         imgtk = ImageTk.PhotoImage(image=img)
#         self.video_label.imgtk = imgtk
#         self.video_label.configure(image=imgtk)
#         self.root.after(15, self.update_translate)

#     def save_transcript(self):
#         txt = self.transcript.get("1.0", tk.END).strip()
#         if not txt:
#             return
#         path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text Files", "*.txt")])
#         if path:
#             with open(path, "w", encoding="utf-8") as f:
#                 f.write(txt)
#             messagebox.showinfo("Saved", f"Transcript saved to:\n{path}")

#     def speak(self):
#         txt = self.transcript.get("1.0", tk.END).strip()
#         if not txt:
#             return
#         self.engine.say(txt)
#         self.engine.runAndWait()

#     # ---------------- Learning Mode ----------------
#     def init_learning_ui(self):
#         self.clear_root()

#         ttk.Label(self.root, text="📚 Learning Mode", font=("Segoe UI", 18, "bold")).pack(pady=10)

#         video_files = [f for f in os.listdir(VIDEO_FOLDER) if f.endswith(".mp4")]
#         if not video_files:
#             ttk.Label(self.root, text="No videos found in 'video' folder.", font=("Segoe UI", 12)).pack(pady=20)
#             ttk.Button(self.root, text="⬅ Back", command=self.show_main_menu).pack(pady=10)
#             return

#         self.video_var = tk.StringVar(value=video_files[0])
#         ttk.Combobox(self.root, textvariable=self.video_var, values=video_files, state="readonly").pack(pady=10)

#         self.video_label = tk.Label(self.root, bg="black")
#         self.video_label.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

#         controls = ttk.Frame(self.root)
#         controls.pack(pady=10)

#         ttk.Button(controls, text="▶ Play", command=self.play_video).pack(side=tk.LEFT, padx=5)
#         ttk.Button(controls, text="⏹ Stop", command=self.stop_video).pack(side=tk.LEFT, padx=5)
#         ttk.Button(controls, text="⬅ Back", command=self.back_from_learning).pack(side=tk.LEFT, padx=5)

#     def play_video(self):
#         video_path = os.path.join(VIDEO_FOLDER, self.video_var.get())
#         if self.video_cap:
#             self.video_cap.release()
#         self.video_cap = cv2.VideoCapture(video_path)
#         self.update_video()

#     def stop_video(self):
#         if self.video_cap:
#             self.video_cap.release()
#             self.video_cap = None
#         self.video_label.config(image='')

#     def update_video(self):
#         if not self.video_cap or not self.video_cap.isOpened():
#             return
#         ret, frame = self.video_cap.read()
#         if not ret:
#             self.stop_video()
#             return
#         frame = cv2.resize(frame, (800, 600))
#         img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#         imgtk = ImageTk.PhotoImage(image=img)
#         self.video_label.imgtk = imgtk
#         self.video_label.configure(image=imgtk)
#         self.root.after(30, self.update_video)

#     def back_from_learning(self):
#         self.stop_video()
#         self.show_main_menu()


# if __name__ == "__main__":
#     root = tk.Tk()
#     app = SignTranslatorApp(root)
#     root.mainloop()

# import os
# import cv2
# import numpy as np
# import pandas as pd
# import mediapipe as mp
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# import joblib
# import tkinter as tk
# from tkinter import ttk, messagebox, filedialog
# from PIL import Image, ImageTk
# import pyttsx3
# from collections import deque, Counter


# # Paths
# DATA_PATH = "sign_data.csv"
# MODEL_PATH = "sign_model.pkl"
# VIDEO_FOLDER = "video"


# # Ensure video folder exists
# if not os.path.exists(VIDEO_FOLDER):
#     os.makedirs(VIDEO_FOLDER)


# # Mediapipe
# mp_hands = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils


# # UI Colors & Fonts
# BG_COLOR = "#f0f4f8"  # Light pastel background
# BTN_COLOR = "#2563eb"  # Bright blue
# BTN_HOVER_COLOR = "#1e4ecc"  # Darker blue on hover
# TEXT_COLOR = "#202324"  # Dark text
# ACCENT_COLOR = "#16a34a"  # Green accent
# FONT_FAMILY = "Segoe UI"


# def normalize_landmarks(landmarks):
#     pts = np.array([(lm.x, lm.y) for lm in landmarks], dtype=np.float32)
#     origin = pts[0]
#     pts -= origin
#     scale = np.max(np.linalg.norm(pts, axis=1)) + 1e-6
#     pts /= scale
#     return pts.flatten()


# class CustomButton(ttk.Button):
#     def __init__(self, master=None, **kw):
#         ttk.Button.__init__(self, master=master, **kw)
#         self.default_style = kw.get('style', 'TButton')
#         self.bind("<Enter>", self.on_enter)
#         self.bind("<Leave>", self.on_leave)

#     def on_enter(self, e):
#         self.configure(style="Hover.TButton")

#     def on_leave(self, e):
#         self.configure(style=self.default_style)


# class SignTranslatorApp:
#     def __init__(self, root):
#         self.root = root
#         self.root.title("🖐 Sign Language Translator")

#         # Set window icon with logo.png (supports transparent PNG)
#         logo_img = Image.open("logo.png")
#         logo_icon = ImageTk.PhotoImage(logo_img)
#         self.root.iconphoto(True, logo_icon)
#         self.logo_icon = logo_icon

#         self.root.geometry("1050x800")
#         self.root.configure(bg=BG_COLOR)

#         # Setup styles
#         self.style = ttk.Style()
#         self.style.theme_use("clam")

#         self.style.configure("TButton",
#                              font=(FONT_FAMILY, 11, "bold"),
#                              padding=10,
#                              relief="flat",
#                              foreground="white",
#                              background=BTN_COLOR)
#         self.style.map("TButton",
#                        background=[('active', BTN_HOVER_COLOR)])

#         self.style.configure("Hover.TButton",
#                              background=BTN_HOVER_COLOR,
#                              foreground="white")

#         self.style.configure("Big.TButton",
#                              font=(FONT_FAMILY, 13, "bold"),
#                              padding=15,
#                              relief="flat",
#                              background=BTN_COLOR,
#                              foreground="white",
#                              borderwidth=0)
#         self.style.map("Big.TButton",
#                        background=[("active", BTN_HOVER_COLOR)])

#         self.style.configure("TLabel",
#                              font=(FONT_FAMILY, 12),
#                              foreground=TEXT_COLOR,
#                              background=BG_COLOR)

#         self.cap = None
#         self.running = False
#         self.hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
#                                     min_detection_confidence=0.5, min_tracking_confidence=0.5)
#         self.engine = pyttsx3.init()
#         self.engine.setProperty("rate", 170)
#         self.clf = None
#         self.labels_ = None
#         self.window = deque(maxlen=12)
#         self.last_prediction = None
#         self.video_cap = None

#         self.show_main_menu()

#     # ---------------- Main Menu ----------------
#     def show_main_menu(self):
#         self.clear_root()

#         frame = ttk.Frame(self.root)
#         frame.pack(expand=True)

#         frame.configure(style="TFrame")

#         logo_img = Image.open("logo.png")
#         logo_img = logo_img.resize((140, 140))
#         self.logo_photo = ImageTk.PhotoImage(logo_img)
#         logo_label = tk.Label(frame, image=self.logo_photo, bg=BG_COLOR)
#         logo_label.pack(pady=(20, 10))

#         ttk.Label(frame, text="🖐 Sign Language Translator",
#                   font=(FONT_FAMILY, 24, "bold"), foreground=ACCENT_COLOR, background=BG_COLOR).pack(pady=20)

#         btn_translate = CustomButton(frame, text="Translate Mode",
#                                      command=self.init_translate_ui, style="Big.TButton")
#         btn_translate.pack(pady=10, ipadx=30, ipady=12, fill=tk.X)

#         btn_learning = CustomButton(frame, text="Learning Mode",
#                                    command=self.init_learning_ui, style="Big.TButton")
#         btn_learning.pack(pady=10, ipadx=30, ipady=12, fill=tk.X)

#         btn_exit = CustomButton(frame, text="Exit", command=self.root.quit, style="Big.TButton")
#         btn_exit.pack(pady=10, ipadx=30, ipady=12, fill=tk.X)

#     def clear_root(self):
#         for widget in self.root.winfo_children():
#             widget.destroy()

#     # ---------------- Translate Mode ----------------
#     def init_translate_ui(self):
#         self.clear_root()

#         self.video_label = tk.Label(self.root, bg="black", bd=4, relief="sunken")
#         self.video_label.pack(padx=15, pady=15, fill=tk.BOTH, expand=True)

#         controls = ttk.Frame(self.root)
#         controls.pack(fill=tk.X, pady=10, padx=15)

#         self.start_btn = CustomButton(controls, text="▶ Start", command=self.start, style="Big.TButton")
#         self.stop_btn = CustomButton(controls, text="⏹ Stop", command=self.stop, style="Big.TButton")
#         self.speak_btn = CustomButton(controls, text="🔊 Speak", command=self.speak, style="Big.TButton")
#         self.reset_btn = CustomButton(controls, text="♻ Reset Transcript", command=self.reset_transcript, style="Big.TButton")
#         self.save_btn = CustomButton(controls, text="💾 Save Transcript", command=self.save_transcript, style="Big.TButton")
#         self.back_btn = CustomButton(controls, text="⬅ Back", command=self.show_main_menu, style="Big.TButton")

#         for btn in [self.start_btn, self.stop_btn, self.speak_btn,
#                     self.reset_btn, self.save_btn, self.back_btn]:
#             btn.pack(side=tk.LEFT, padx=5, pady=3, fill=tk.X, expand=True)

#         status_frame = ttk.Frame(self.root)
#         status_frame.pack(fill=tk.X, pady=8, padx=15)

#         ttk.Label(status_frame, text="Prediction:", font=(FONT_FAMILY, 16, "bold"), foreground=ACCENT_COLOR).pack(side=tk.LEFT, padx=(0, 10))

#         self.pred_var = tk.StringVar(value="—")
#         self.pred_label = ttk.Label(status_frame, textvariable=self.pred_var,
#                                     font=(FONT_FAMILY, 20, "bold"), foreground=ACCENT_COLOR)
#         self.pred_label.pack(side=tk.LEFT)

#         self.conf_var = tk.StringVar(value="")
#         ttk.Label(status_frame, textvariable=self.conf_var, font=(FONT_FAMILY, 14),
#                   foreground="#4b5563").pack(side=tk.LEFT, padx=10)

#         transcript_frame = ttk.LabelFrame(self.root, text="Transcript", padding=15, style="TFrame")
#         transcript_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)

#         self.transcript = tk.Text(transcript_frame, height=8,
#                                   font=(FONT_FAMILY, 14),
#                                   bg="white",
#                                   fg=TEXT_COLOR,
#                                   insertbackground=TEXT_COLOR,
#                                   bd=2,
#                                   relief="groove",
#                                   padx=10, pady=10)
#         self.transcript.pack(fill=tk.BOTH, expand=True)

#         self.prepare_and_train()


#     def prepare_and_train(self):
#         if not os.path.exists(DATA_PATH):
#             messagebox.showerror("Dataset Missing", f"{DATA_PATH} not found.")
#             return
#         df = pd.read_csv(DATA_PATH, header=None)
#         if df.empty:
#             messagebox.showerror("Dataset Empty", f"{DATA_PATH} is empty.")
#             return
#         X = df.iloc[:, :-1].values
#         y = df.iloc[:, -1].values
#         try:
#             Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
#         except Exception:
#             Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
#         self.clf = RandomForestClassifier(n_estimators=300, random_state=42)
#         self.clf.fit(Xtr, ytr)
#         if len(np.unique(yte)) > 1 and len(yte) > 0:
#             yhat = self.clf.predict(Xte)
#             acc = accuracy_score(yte, yhat)
#             self.conf_var.set(f"Model Accuracy: {acc:.2f}")
#         self.labels_ = np.unique(y)
#         joblib.dump(self.clf, MODEL_PATH)

#     def start(self):
#         if self.running:
#             return
#         if self.clf is None:
#             messagebox.showerror("No Model", "Train the model first.")
#             return
#         self.cap = cv2.VideoCapture(0)
#         if not self.cap.isOpened():
#             messagebox.showerror("Camera Error", "Could not open webcam.")
#             return
#         self.running = True
#         self.update_translate()

#     def stop(self):
#         self.running = False
#         if self.cap:
#             self.cap.release()
#         self.video_label.config(image='')
#         self.window.clear()

#     def reset_transcript(self):
#         self.transcript.delete("1.0", tk.END)
#         self.last_prediction = None

#     def update_translate(self):
#         if not self.running:
#             return
#         ok, frame = self.cap.read()
#         if not ok:
#             self.stop()
#             return
#         frame = cv2.flip(frame, 1)
#         rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         res = self.hands.process(rgb)
#         pred = None
#         pred_conf = 0.0
#         if res.multi_hand_landmarks:
#             hand = res.multi_hand_landmarks[0]
#             mp_drawing.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
#             feat = normalize_landmarks(hand.landmark)
#             if hasattr(self.clf, "predict_proba"):
#                 proba = self.clf.predict_proba([feat])[0]
#                 pred_idx = int(np.argmax(proba))
#                 pred = self.clf.classes_[pred_idx]
#                 pred_conf = float(proba[pred_idx])
#             else:
#                 pred = self.clf.predict([feat])[0]
#                 pred_conf = 1.0
#             self.window.append(pred)
#             if len(self.window) == self.window.maxlen:
#                 common, count = Counter(self.window).most_common(1)[0]
#                 pred = common
#                 pred_conf = count / len(self.window)
#         if pred is not None:
#             self.pred_var.set(str(pred))
#             if pred_conf >= 0.7 and pred != self.last_prediction:
#                 self.transcript.insert(tk.END, pred + ' ')
#                 self.transcript.see(tk.END)
#                 self.last_prediction = pred
#         else:
#             self.pred_var.set("—")
#             self.last_prediction = None
#         img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#         imgtk = ImageTk.PhotoImage(image=img)
#         self.video_label.imgtk = imgtk
#         self.video_label.configure(image=imgtk)
#         self.root.after(15, self.update_translate)

#     def save_transcript(self):
#         txt = self.transcript.get("1.0", tk.END).strip()
#         if not txt:
#             return
#         path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text Files", "*.txt")])
#         if path:
#             with open(path, "w", encoding="utf-8") as f:
#                 f.write(txt)
#             messagebox.showinfo("Saved", f"Transcript saved to:\n{path}")

#     def speak(self):
#         txt = self.transcript.get("1.0", tk.END).strip()
#         if not txt:
#             return
#         self.engine.say(txt)
#         self.engine.runAndWait()

#     def on_closing(self):
#         if messagebox.askokcancel("Exit", "Are you sure you want to exit?"):
#             self.stop()
#             self.root.destroy()

#     # ---------------- Learning Mode ----------------
#     def init_learning_ui(self):
#         self.clear_root()

#         ttk.Label(self.root, text="📚 Learning Mode",
#                   font=(FONT_FAMILY, 20, "bold"), foreground=ACCENT_COLOR).pack(pady=15)

#         video_files = [f for f in os.listdir(VIDEO_FOLDER) if f.endswith(".mp4")]
#         if not video_files:
#             ttk.Label(self.root, text="No videos found in 'video' folder.",
#                       font=(FONT_FAMILY, 14), background=BG_COLOR).pack(pady=20)
#             btn_back = CustomButton(self.root, text="⬅ Back", command=self.show_main_menu, style="Big.TButton")
#             btn_back.pack(pady=10, ipadx=30, ipady=12)
#             return

#         self.video_var = tk.StringVar(value=video_files[0])
#         combo = ttk.Combobox(self.root, textvariable=self.video_var,
#                              values=video_files, state="readonly", font=(FONT_FAMILY, 12))
#         combo.pack(pady=15, padx=15)

#         self.video_label = tk.Label(self.root, bg="black", bd=4, relief="sunken")
#         self.video_label.pack(padx=15, pady=15, fill=tk.BOTH, expand=True)

#         controls = ttk.Frame(self.root)
#         controls.pack(pady=10, padx=15)

#         btn_play = CustomButton(controls, text="▶ Play", command=self.play_video, style="Big.TButton")
#         btn_stop = CustomButton(controls, text="⏹ Stop", command=self.stop_video, style="Big.TButton")
#         btn_back2 = CustomButton(controls, text="⬅ Back", command=self.back_from_learning, style="Big.TButton")

#         for btn in [btn_play, btn_stop, btn_back2]:
#             btn.pack(side=tk.LEFT, padx=5, pady=3, fill=tk.X, expand=True)

#     def play_video(self):
#         video_path = os.path.join(VIDEO_FOLDER, self.video_var.get())
#         if self.video_cap:
#             self.video_cap.release()
#         self.video_cap = cv2.VideoCapture(video_path)
#         self.update_video()

#     def stop_video(self):
#         if self.video_cap:
#             self.video_cap.release()
#             self.video_cap = None
#         self.video_label.config(image='')

#     def update_video(self):
#         if not self.video_cap or not self.video_cap.isOpened():
#             return
#         ret, frame = self.video_cap.read()
#         if not ret:
#             self.stop_video()
#             return
#         frame = cv2.resize(frame, (800, 600))
#         img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#         imgtk = ImageTk.PhotoImage(image=img)
#         self.video_label.imgtk = imgtk
#         self.video_label.configure(image=imgtk)
#         self.root.after(30, self.update_video)

#     def back_from_learning(self):
#         self.stop_video()
#         self.show_main_menu()


# if __name__ == "__main__":
#     root = tk.Tk()
#     app = SignTranslatorApp(root)
#     root.protocol("WM_DELETE_WINDOW", app.on_closing)
#     root.mainloop()



import os
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
import pyttsx3
from collections import deque, Counter


# Paths
DATA_PATH = "sign_data.csv"
MODEL_PATH = "sign_model.pkl"
VIDEO_FOLDER = "video"


# Ensure video folder exists
if not os.path.exists(VIDEO_FOLDER):
    os.makedirs(VIDEO_FOLDER)


# Mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


# UI Colors
BG_COLOR = "#ffffff"
BTN_COLOR = "#3b82f6"
TEXT_COLOR = "#000000"
ACCENT_COLOR = "#16a34a"


def normalize_landmarks(landmarks):
    pts = np.array([(lm.x, lm.y) for lm in landmarks], dtype=np.float32)
    origin = pts[0]
    pts -= origin
    scale = np.max(np.linalg.norm(pts, axis=1)) + 1e-6
    pts /= scale
    return pts.flatten()


class SignTranslatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🖐 Sign Language Translator")

        # Set window icon with logo.png (supports transparent PNG)
        logo_img = Image.open("logo.png")
        logo_icon = ImageTk.PhotoImage(logo_img)
        self.root.iconphoto(True, logo_icon)
        self.logo_icon = logo_icon  # Prevent garbage collection

        self.root.geometry("1050x800")
        self.root.configure(bg=BG_COLOR)

        # Common variables
        self.cap = None
        self.running = False
        self.hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                                    min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", 170)
        self.clf = None
        self.labels_ = None
        self.window = deque(maxlen=12)
        self.last_prediction = None
        self.video_cap = None

        self.show_main_menu()

    # ---------------- Main Menu ----------------
    def show_main_menu(self):
        self.clear_root()

        frame = ttk.Frame(self.root)
        frame.pack(expand=True)

        # Display Logo above the title
        logo_img = Image.open("logo.png")
        logo_img = logo_img.resize((120, 120))  # Adjust size as needed
        self.logo_photo = ImageTk.PhotoImage(logo_img)
        logo_label = tk.Label(frame, image=self.logo_photo, bg=BG_COLOR)
        logo_label.pack(pady=(18, 12))

        ttk.Label(frame, text="🖐 Sign Language Translator", font=("Segoe UI", 20, "bold")).pack(pady=20)

        ttk.Button(frame, text="Translate Mode", command=self.init_translate_ui,
                   style="TButton").pack(pady=10, ipadx=20, ipady=10)
        ttk.Button(frame, text="Learning Mode", command=self.init_learning_ui,
                   style="TButton").pack(pady=10, ipadx=20, ipady=10)
        ttk.Button(frame, text="Exit", command=self.root.quit,
                   style="TButton").pack(pady=10, ipadx=20, ipady=10)

    def clear_root(self):
        for widget in self.root.winfo_children():
            widget.destroy()

    # ---------------- Translate Mode ----------------
    def init_translate_ui(self):
        self.clear_root()

        self.video_label = tk.Label(self.root, bg="black")
        self.video_label.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        controls = ttk.Frame(self.root)
        controls.pack(fill=tk.X, pady=10)

        self.start_btn = ttk.Button(controls, text="▶ Start", command=self.start)
        self.stop_btn = ttk.Button(controls, text="⏹ Stop", command=self.stop)
        self.speak_btn = ttk.Button(controls, text="🔊 Speak", command=self.speak)
        self.reset_btn = ttk.Button(controls, text="♻ Reset Transcript", command=self.reset_transcript)
        self.save_btn = ttk.Button(controls, text="💾 Save Transcript", command=self.save_transcript)
        self.back_btn = ttk.Button(controls, text="⬅ Back", command=self.show_main_menu)

        for btn in [self.start_btn, self.stop_btn, self.speak_btn,
                    self.reset_btn, self.save_btn, self.back_btn]:
            btn.pack(side=tk.LEFT, padx=3, pady=3, fill=tk.X, expand=True)

        status_frame = ttk.Frame(self.root)
        status_frame.pack(fill=tk.X, pady=5)
        ttk.Label(status_frame, text="Prediction:", font=("Segoe UI", 14, "bold")).pack(side=tk.LEFT, padx=5)
        self.pred_var = tk.StringVar(value="—")
        self.pred_label = ttk.Label(status_frame, textvariable=self.pred_var,
                                    font=("Segoe UI", 18, "bold"), foreground=ACCENT_COLOR)
        self.pred_label.pack(side=tk.LEFT, padx=10)
        self.conf_var = tk.StringVar(value="")
        ttk.Label(status_frame, textvariable=self.conf_var, font=("Segoe UI", 12),
                  foreground="#4b5563").pack(side=tk.LEFT)

        transcript_frame = ttk.LabelFrame(self.root, text="Transcript")
        transcript_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.transcript = tk.Text(transcript_frame, height=8, font=("Segoe UI", 12),
                                 bg="#f3f4f6", fg=TEXT_COLOR, insertbackground="black")
        self.transcript.pack(fill=tk.BOTH, expand=True)

        self.prepare_and_train()

    def prepare_and_train(self):
        if not os.path.exists(DATA_PATH):
            messagebox.showerror("Dataset Missing", f"{DATA_PATH} not found.")
            return
        df = pd.read_csv(DATA_PATH, header=None)
        if df.empty:
            messagebox.showerror("Dataset Empty", f"{DATA_PATH} is empty.")
            return
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        try:
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        except Exception:
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
        self.clf = RandomForestClassifier(n_estimators=300, random_state=42)
        self.clf.fit(Xtr, ytr)
        if len(np.unique(yte)) > 1 and len(yte) > 0:
            yhat = self.clf.predict(Xte)
            acc = accuracy_score(yte, yhat)
            self.conf_var.set(f"Model Accuracy: {acc:.2f}")
        self.labels_ = np.unique(y)
        joblib.dump(self.clf, MODEL_PATH)

    def start(self):
        if self.running:
            return
        if self.clf is None:
            messagebox.showerror("No Model", "Train the model first.")
            return
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Camera Error", "Could not open webcam.")
            return
        self.running = True
        self.update_translate()

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.video_label.config(image='')
        self.window.clear()

    def reset_transcript(self):
        self.transcript.delete("1.0", tk.END)
        self.last_prediction = None

    def update_translate(self):
        if not self.running:
            return
        ok, frame = self.cap.read()
        if not ok:
            self.stop()
            return
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.hands.process(rgb)
        pred = None
        pred_conf = 0.0
        if res.multi_hand_landmarks:
            hand = res.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
            feat = normalize_landmarks(hand.landmark)
            if hasattr(self.clf, "predict_proba"):
                proba = self.clf.predict_proba([feat])[0]
                pred_idx = int(np.argmax(proba))
                pred = self.clf.classes_[pred_idx]
                pred_conf = float(proba[pred_idx])
            else:
                pred = self.clf.predict([feat])[0]
                pred_conf = 1.0
            self.window.append(pred)
            if len(self.window) == self.window.maxlen:
                common, count = Counter(self.window).most_common(1)[0]
                pred = common
                pred_conf = count / len(self.window)
        if pred is not None:
            self.pred_var.set(str(pred))
            if pred_conf >= 0.7 and pred != self.last_prediction:
                self.transcript.insert(tk.END, pred + ' ')
                self.transcript.see(tk.END)
                self.last_prediction = pred
        else:
            self.pred_var.set("—")
            self.last_prediction = None
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)
        self.root.after(15, self.update_translate)

    def save_transcript(self):
        txt = self.transcript.get("1.0", tk.END).strip()
        if not txt:
            return
        path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text Files", "*.txt")])
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(txt)
            messagebox.showinfo("Saved", f"Transcript saved to:\n{path}")

    def speak(self):
        txt = self.transcript.get("1.0", tk.END).strip()
        if not txt:
            return
        self.engine.say(txt)
        self.engine.runAndWait()

    def on_closing(self):
        if messagebox.askokcancel("Exit", "Are you sure you want to exit?"):
            self.stop()
            self.root.destroy()

    # ---------------- Learning Mode ----------------
    def init_learning_ui(self):
        self.clear_root()

        ttk.Label(self.root, text="📚 Learning Mode", font=("Segoe UI", 18, "bold")).pack(pady=10)

        video_files = [f for f in os.listdir(VIDEO_FOLDER) if f.endswith(".mp4")]
        if not video_files:
            ttk.Label(self.root, text="No videos found in 'video' folder.", font=("Segoe UI", 12)).pack(pady=20)
            ttk.Button(self.root, text="⬅ Back", command=self.show_main_menu).pack(pady=10)
            return

        self.video_var = tk.StringVar(value=video_files[0])
        ttk.Combobox(self.root, textvariable=self.video_var, values=video_files, state="readonly").pack(pady=10)

        self.video_label = tk.Label(self.root, bg="black")
        self.video_label.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        controls = ttk.Frame(self.root)
        controls.pack(pady=10)

        ttk.Button(controls, text="▶ Play", command=self.play_video).pack(side=tk.LEFT, padx=5)
        ttk.Button(controls, text="⏹ Stop", command=self.stop_video).pack(side=tk.LEFT, padx=5)
        ttk.Button(controls, text="⬅ Back", command=self.back_from_learning).pack(side=tk.LEFT, padx=5)

    def play_video(self):
        video_path = os.path.join(VIDEO_FOLDER, self.video_var.get())
        if self.video_cap:
            self.video_cap.release()
        self.video_cap = cv2.VideoCapture(video_path)
        self.update_video()

    def stop_video(self):
        if self.video_cap:
            self.video_cap.release()
            self.video_cap = None
        self.video_label.config(image='')

    def update_video(self):
        if not self.video_cap or not self.video_cap.isOpened():
            return
        ret, frame = self.video_cap.read()
        if not ret:
            self.stop_video()
            return
        frame = cv2.resize(frame, (800, 600))
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)
        self.root.after(30, self.update_video)

    def back_from_learning(self):
        self.stop_video()
        self.show_main_menu()


if __name__ == "__main__":
    root = tk.Tk()
    app = SignTranslatorApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
