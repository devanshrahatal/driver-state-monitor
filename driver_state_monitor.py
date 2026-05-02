"""
DriveGuard — Driver State Monitoring System
───────────────────────────────────────────
Real-time computer vision system that monitors driver attentiveness 
using MediaPipe Face Mesh. Detects micro-sleep, fatigue, and distraction,
and initiates emergency protocols (audio alarms, TTS, email dispatch) 
if critical unresponsiveness is detected.

Detailed feature lists and usage instructions are maintained in README.md.
"""

import cv2
import numpy as np
from scipy.spatial import distance as dist
import time
import csv
from collections import deque
import threading
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
import sys
import json
import subprocess
import platform
from dotenv import load_dotenv

load_dotenv()

# Text-to-speech (non-blocking)
try:
    import pyttsx3 as _pyttsx3
    def speak(text):
        def _run():
            global alarm_sound
            try:
                engine = _pyttsx3.init()
                engine.setProperty('rate', 165)
                # Audio ducking: lower alarm volume during speech
                if 'alarm_sound' in globals() and alarm_sound is not None:
                    try:
                        alarm_sound.set_volume(0.15)
                    except Exception:
                        pass
                
                engine.say(text)
                engine.runAndWait()
            except Exception:
                pass
            finally:
                # Restore volume
                if 'alarm_sound' in globals() and alarm_sound is not None:
                    try:
                        alarm_sound.set_volume(1.0)
                    except Exception:
                        pass
        threading.Thread(target=_run, daemon=True).start()
except ImportError:
    def speak(text):   # silent fallback if pyttsx3 not installed
        pass

# ── Premium UI Palette (BGR for OpenCV) ───────────────────────────────────────
COLOR_BG       = (36,  22,  18)    # Deep navy background
COLOR_PANEL    = (52,  33,  28)    # Card / panel bg
COLOR_PANEL2   = (64,  42,  36)    # Slightly lighter card
COLOR_BORDER   = (95,  65,  55)    # Subtle border
COLOR_TEXT     = (245, 228, 220)   # Soft white text
COLOR_SUBTEXT  = (170, 135, 120)   # Dimmed subtext
COLOR_ACCENT   = (255, 200,  80)   # Cyan-blue accent
COLOR_WARN     = (255, 200,  80)   # Cyan-blue (matching accent)
COLOR_YEL      = (255, 200,  80)   # Cyan-blue (matching accent)
COLOR_DANGER   = (255,  80,  60)   # Electric red (drowsy)
COLOR_RED      = (240,  80,  50)   # Pure alert red
COLOR_OK       = (140, 220,  60)   # Bright green (active)
COLOR_PURP     = (240,  80, 200)   # Purple (distracted)
COLOR_DARK_BAR = (62,   41,  35)   # Progress bar track

# ── Drawing Helpers ───────────────────────────────────────────────────────────

def draw_rounded_rect(img, x1, y1, x2, y2, r, color, thickness=-1):
    """Draw a filled or outlined rounded rectangle."""
    if thickness == -1:  # filled
        cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1)
        for cx, cy in [(x1+r, y1+r), (x2-r, y1+r), (x1+r, y2-r), (x2-r, y2-r)]:
            cv2.circle(img, (cx, cy), r, color, -1)
    else:
        cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, thickness)
        cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, thickness)
        for cx, cy in [(x1+r, y1+r), (x2-r, y1+r), (x1+r, y2-r), (x2-r, y2-r)]:
            cv2.ellipse(img, (cx, cy), (r, r), 0, 0, 0, color, thickness)
            cv2.ellipse(img, (cx, cy), (r, r), 0, 90, 90, color, thickness)
            cv2.ellipse(img, (cx, cy), (r, r), 0, 180, 180, color, thickness)
            cv2.ellipse(img, (cx, cy), (r, r), 0, 270, 270, color, thickness)

def draw_pill(img, x1, y1, x2, y2, color):
    """Filled pill / capsule shape."""
    r = (y2 - y1) // 2
    cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1)
    cv2.circle(img, (x1 + r, (y1 + y2) // 2), r, color, -1)
    cv2.circle(img, (x2 - r, (y1 + y2) // 2), r, color, -1)

def draw_section_header(img, x, y, label, accent_color=None):
    """Small left accent bar + uppercase section label."""
    if accent_color is None:
        accent_color = COLOR_ACCENT
    cv2.rectangle(img, (x, y - 12), (x + 3, y + 4), accent_color, -1)
    cv2.putText(img, label, (x + 9, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                COLOR_SUBTEXT, 1, cv2.LINE_AA)

def draw_bar(image, x, y, w, h, value, max_value, color):
    """Rounded progress bar with track."""
    r = h // 2
    # Track
    draw_pill(image, x, y, x + w, y + h, COLOR_DARK_BAR)
    # Fill
    val_clamped = max(0, min(value, max_value))
    fill_w = int((val_clamped / max_value) * w)
    if fill_w > h:  # ensure pill looks right
        draw_pill(image, x, y, x + fill_w, y + h, color)
    elif fill_w > 0:
        cv2.circle(image, (x + r, y + r), r, color, -1)

def blend_rect(img, x1, y1, x2, y2, color, alpha=0.55):
    """Semi-transparent filled rectangle overlay."""
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

# ══════════════════════════════════════════════════════════════════════════════
#  STARTUP: RECEIVER EMAIL PROMPT
# ══════════════════════════════════════════════════════════════════════════════

def show_email_prompt():
    """Show a styled startup window to collect the emergency receiver email."""
    import tkinter as tk

    try:
        from tkinterdnd2 import TkinterDnD
        root = TkinterDnD.Tk()
    except ImportError:
        root = tk.Tk()

    # Colour palette (matching DriveGuard dashboard)
    BG            = "#121624"
    PANEL2        = "#242840"
    BORDER_CLR    = "#37415F"
    TEXT_CLR      = "#DCE4F5"
    SUBTEXT_CLR   = "#7887AA"
    ACCENT_CLR    = "#50C8FF"
    ACCENT_HOVER  = "#6DD4FF"
    DANGER_CLR    = "#F05050"

    root.title("DriveGuard \u2014 Setup")
    root.configure(bg=BG)
    root.resizable(False, False)

    win_w, win_h = 520, 310
    sw = root.winfo_screenwidth()
    sh = root.winfo_screenheight()
    root.geometry(f"{win_w}x{win_h}+{(sw - win_w) // 2}+{(sh - win_h) // 2}")

    # ── Header bar ───────────────────────────────────────────────────────────
    hdr = tk.Frame(root, bg=PANEL2, height=50)
    hdr.pack(fill="x")
    hdr.pack_propagate(False)

    tk.Label(hdr, text="  DG  ", bg=ACCENT_CLR, fg="#0A0F19",
             font=("Segoe UI", 14, "bold")).pack(side="left", fill="y")
    tk.Label(hdr, text="DriveGuard", bg=PANEL2, fg=TEXT_CLR,
             font=("Segoe UI", 13, "bold")).pack(side="left", padx=(10, 0))
    tk.Label(hdr, text="Setup", bg=PANEL2, fg=SUBTEXT_CLR,
             font=("Segoe UI", 9)).pack(side="left", padx=(8, 0))

    tk.Frame(root, bg=ACCENT_CLR, height=2).pack(fill="x")

    # ── Body ─────────────────────────────────────────────────────────────────
    body = tk.Frame(root, bg=BG)
    body.pack(fill="both", expand=True, padx=30, pady=18)

    tk.Label(body, text="\u26a0", bg=BG, fg=DANGER_CLR,
             font=("Segoe UI", 24, "bold")).pack()
    tk.Label(body, text="Emergency Contact Setup",
             bg=BG, fg=TEXT_CLR,
             font=("Segoe UI", 13, "bold")).pack(pady=(4, 2))
    tk.Label(body,
             text="Enter the email address to receive emergency alerts",
             bg=BG, fg=SUBTEXT_CLR,
             font=("Segoe UI", 9)).pack(pady=(0, 14))

    # ── Email entry ──────────────────────────────────────────────────────────
    ef = tk.Frame(body, bg=BG)
    ef.pack(fill="x")

    tk.Label(ef, text="Receiver Email:", bg=BG, fg=SUBTEXT_CLR,
             font=("Segoe UI", 9)).pack(anchor="w")

    email_var = tk.StringVar(value="")
    email_entry = tk.Entry(ef, textvariable=email_var, bg="#0E1220",
                           fg=TEXT_CLR, font=("Consolas", 11),
                           insertbackground=ACCENT_CLR, relief="flat",
                           highlightbackground=BORDER_CLR,
                           highlightthickness=1, highlightcolor=ACCENT_CLR)
    email_entry.pack(fill="x", pady=(4, 0), ipady=6)
    email_entry.focus_set()

    # Status line
    status_var = tk.StringVar(value="")
    status_lbl = tk.Label(body, textvariable=status_var, bg=BG,
                          fg=DANGER_CLR, font=("Segoe UI", 9))
    status_lbl.pack(anchor="w", pady=(6, 0))

    result = {"email": None}

    def on_start():
        email = email_var.get().strip()
        if not email or "@" not in email or "." not in email.split("@")[-1]:
            status_var.set("\u26a0  Please enter a valid email address.")
            return
        result["email"] = email
        root.destroy()

    def on_close():
        root.destroy()
        sys.exit(0)

    root.protocol("WM_DELETE_WINDOW", on_close)
    email_entry.bind("<Return>", lambda e: on_start())

    # ── Start button ─────────────────────────────────────────────────────────
    bf = tk.Frame(body, bg=BG)
    bf.pack(pady=(14, 0))

    start_btn = tk.Button(bf, text="  Start Monitoring  ",
                          bg=ACCENT_CLR, fg="#0A0F19",
                          font=("Segoe UI", 11, "bold"), relief="flat",
                          cursor="hand2", activebackground=ACCENT_HOVER,
                          activeforeground="#0A0F19", command=on_start,
                          bd=0, padx=24, pady=8)
    start_btn.pack()

    start_btn.bind("<Enter>", lambda e: start_btn.config(bg=ACCENT_HOVER))
    start_btn.bind("<Leave>", lambda e: start_btn.config(bg=ACCENT_CLR))

    root.mainloop()
    return result["email"]


_startup_receiver_email = show_email_prompt()
if _startup_receiver_email is None:
    sys.exit(0)

# Open Camera FIRST (fastest to start)
cap = cv2.VideoCapture(0)

# Background initialization for heavy modules
face_mesh = None
alarm_sound = None
system_ready = False

def init_heavy_modules():
    global face_mesh, alarm_sound, system_ready
    import mediapipe as mp
    import pygame
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
    pygame.mixer.init()
    alarm_sound = pygame.mixer.Sound("alarm.wav")
    system_ready = True

init_thread = threading.Thread(target=init_heavy_modules, daemon=True)
init_thread.start()

# Landmark indices (MediaPipe Face Mesh)
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
NOSE_TIP = 1
LEFT_CHEEK = 234
RIGHT_CHEEK = 454

# Head Pose Estimation landmarks
POSE_LANDMARKS = [1, 33, 263, 61, 291, 199]

# Calculate Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# Calculate Mouth Aspect Ratio (MAR)
def mouth_aspect_ratio(mouth):
    vertical = dist.euclidean(mouth[0], mouth[1])
    horizontal = dist.euclidean(mouth[2], mouth[3])
    return vertical / horizontal

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION LOADER
# ══════════════════════════════════════════════════════════════════════════════

CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")

def load_config():
    """Load settings from config.json, falling back to defaults if missing."""
    defaults = {
        "thresholds": {
            "ear_threshold": 0.23, "drowsy_time": 2.0, "sleepy_blink_time": 0.3,
            "blink_rate_threshold": 25, "recovery_time": 10, "mar_threshold": 0.6,
            "yawn_time": 2.0, "distract_time": 2.0, "distract_recovery_time": 0.5,
            "no_face_threshold": 1.5, "no_face_distracted_threshold": 3.0,
        },
        "emergency": {
            "emergency_time": 15.0, "repeat_interval": 30.0,
        },
        "session": {
            "break_reminder_minutes": 90, "record_duration": 5, "panel_width": 550,
        },
    }
    if os.path.isfile(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                user_cfg = json.load(f)
            # Merge: user values override defaults
            for section in defaults:
                if section in user_cfg:
                    defaults[section].update(user_cfg[section])
        except Exception as e:
            print(f"[WARN] Could not load config.json: {e} — using defaults.")
    else:
        print("[INFO] config.json not found — using defaults.")
    return defaults

CFG = load_config()

# Configuration Thresholds (from config)
EAR_THRESHOLD = CFG["thresholds"]["ear_threshold"]
DROWSY_TIME = CFG["thresholds"]["drowsy_time"]
SLEEPY_BLINK_TIME = CFG["thresholds"]["sleepy_blink_time"]
BLINK_RATE_THRESHOLD = CFG["thresholds"]["blink_rate_threshold"]
RECOVERY_TIME = CFG["thresholds"]["recovery_time"]
MAR_THRESHOLD = CFG["thresholds"]["mar_threshold"]
YAWN_TIME = CFG["thresholds"]["yawn_time"]

DISTRACT_TIME = CFG["thresholds"]["distract_time"]
DISTRACT_RECOVERY_TIME = CFG["thresholds"]["distract_recovery_time"]
PANEL_WIDTH = CFG["session"]["panel_width"]

# Emergency Email Settings (from config + .env)
EMERGENCY_TIME = CFG["emergency"]["emergency_time"]
SENDER_EMAIL = os.getenv("DRIVEGUARD_SENDER_EMAIL", "")
SENDER_PASSWORD = os.getenv("DRIVEGUARD_SENDER_PASSWORD", "")
RECEIVER_EMAIL = _startup_receiver_email  # Set by startup prompt

EMERGENCY_REPEAT_INTERVAL = CFG["emergency"]["repeat_interval"]

def send_emergency_email():
    try:
        current_timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        
        msg = MIMEMultipart()
        msg["From"] = SENDER_EMAIL
        msg["To"] = RECEIVER_EMAIL
        msg["Subject"] = f"\U0001f6a8 EMERGENCY: Driver Unresponsive! [{current_timestamp}]"

        body = f"""EMERGENCY ALERT!

The driver has been unresponsive (eyes closed) for more than 15 seconds.
Please check on them immediately!

Alert Time: {current_timestamp}

This is an automated alert from the Driver Monitoring System.
Alerts will continue every 30 seconds until the driver responds."""

        msg.attach(MIMEText(body, "plain"))

        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.sendmail(SENDER_EMAIL, RECEIVER_EMAIL, msg.as_string())
        server.quit()

        print(f"[{current_timestamp}] Emergency email sent to {RECEIVER_EMAIL} successfully!")
    except Exception as e:
        print(f"Failed to send emergency email: {e}")

# Runtime Variables
last_emergency_email_time = None
email_toast_time = None          # Timestamp when last email was sent (for toast)
eye_closed_start = None
alarm_on = False
blink_times = deque()
sleepy_start_time = None

yawn_start_time = None
yawn_count = 0

state = "ACTIVE"
state_color = (0, 255, 0)

distraction_start_time = None
distract_recovery_start = None

# Face Detection Buffer
face_missing_start = None
NO_FACE_THRESHOLD = CFG["thresholds"]["no_face_threshold"]
NO_FACE_DISTRACTED_THRESHOLD = CFG["thresholds"]["no_face_distracted_threshold"]

# Fatigue Score Tracker
fatigue_score = 0.0
fatigue_level = "SAFE"
last_fatigue_update = time.time()
drowsy_active = False
last_logged_state = None
last_csv_log_time = 0  # Timer for 1-second interval CSV logging

session_timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")

# Video Evidence Settings
EVIDENCE_DIR = "evidence"
if not os.path.exists(EVIDENCE_DIR):
    os.makedirs(EVIDENCE_DIR)

session_evidence_dir = os.path.join(EVIDENCE_DIR, f"session_{session_timestamp}")
if not os.path.exists(session_evidence_dir):
    os.makedirs(session_evidence_dir)

recording = False
video_writer = None
record_start_time = None
RECORD_DURATION = CFG["session"]["record_duration"]

# Session Timer & Break Reminder
session_start_time = time.time()
BREAK_REMINDER_SEC = CFG["session"]["break_reminder_minutes"] * 60
last_break_reminder_time = 0      # Epoch of last break reminder spoken
break_reminder_shown = False      # Whether the dashboard banner is active
break_banner_time = None          # When to hide the banner

# CSV Logging Setup - New file per session in records folder
RECORDS_DIR = "records"
if not os.path.exists(RECORDS_DIR):
    os.makedirs(RECORDS_DIR)

LOG_FILE = os.path.join(RECORDS_DIR, f"session_{session_timestamp}.csv")
with open(LOG_FILE, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Date", "Time", "State", "FatigueScore", "EAR", "BlinkRate", "YawnCount", "HeadDirection", "NoseOffset", "MAR"])

# Camera already opened at startup

# Calculate fatigue based on weighted factors
def update_fatigue(score, state, blink_rate, yawn_event, distracted, drowsy_active, dt):
    # Increase score based on active 
    if drowsy_active:
        score += 40 * dt
    elif state == "SLEEPY":
        score += 20 * dt

    if blink_rate >= 20:
        score += 15 * dt

    if yawn_event:
        score += 10

    if distracted:
        score += 10 * dt

    # Gradual recovery when attentive
    if state == "ACTIVE" and not distracted and not drowsy_active:
        score -= 5 * dt

    score = max(0, min(100, score))
    return score

# Show loading screen while heavy modules initialize
PANEL_WIDTH_LOAD = 550
while not system_ready:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]

    # ── Loading Screen ─────────────────────────────────────────────────────────
    combined_display = np.zeros((h, w + PANEL_WIDTH_LOAD, 3), dtype=np.uint8)
    combined_display[:] = COLOR_BG
    combined_display[:h, :w] = frame

    # Camera dim overlay
    blend_rect(combined_display, 0, 0, w, h, (0, 0, 0), alpha=0.55)

    # Animated spinner (arc)
    cx_s, cy_s = w // 2, h // 2 - 30
    spin_angle = int(time.time() * 300) % 360
    for i in range(6):
        a = spin_angle + i * 50
        rad = np.radians(a)
        sx = int(cx_s + 34 * np.cos(rad))
        sy = int(cy_s + 34 * np.sin(rad))
        alpha_c = max(40, 255 - i * 35)
        col = tuple(int(c * alpha_c / 255) for c in COLOR_ACCENT)
        cv2.circle(combined_display, (sx, sy), 5 - i // 2, col, -1)

    # Loading text
    text = "INITIALIZING"
    ts = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
    tx = (w - ts[0]) // 2
    ty = cy_s + 65
    cv2.putText(combined_display, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, COLOR_ACCENT, 2, cv2.LINE_AA)
    # Animated dots
    dots = [".", "..", "...", "...."][int(time.time() * 2) % 4]
    cv2.putText(combined_display, dots, (tx + ts[0] + 4, ty),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR_ACCENT, 2, cv2.LINE_AA)

    # ── Side Panel (Loading) ───────────────────────────────────────────────────
    PW = PANEL_WIDTH_LOAD
    px = w
    # Panel background
    combined_display[:h, px:px+PW] = COLOR_PANEL
    # Header gradient strip
    cv2.rectangle(combined_display, (px, 0), (px + PW, 58), COLOR_PANEL2, -1)
    cv2.rectangle(combined_display, (px, 56), (px + PW, 58), COLOR_ACCENT, -1)

    # Logo area
    cv2.rectangle(combined_display, (px, 0), (px + 58, 58), COLOR_ACCENT, -1)
    cv2.putText(combined_display, "DG", (px + 10, 38),
                cv2.FONT_HERSHEY_DUPLEX, 0.95, (10, 15, 25), 2, cv2.LINE_AA)
    cv2.putText(combined_display, "DriveGuard", (px + 68, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_TEXT, 1, cv2.LINE_AA)
    cv2.putText(combined_display, "Driver Monitoring System", (px + 68, 44),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, COLOR_SUBTEXT, 1, cv2.LINE_AA)

    # Loading status
    draw_rounded_rect(combined_display, px+16, 80, px+PW-16, 130, 6, COLOR_PANEL2)
    cv2.putText(combined_display, "System Startup", (px + 28, 102),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_TEXT, 1, cv2.LINE_AA)
    cv2.putText(combined_display, "Loading Analyzing Metrics...", (px + 28, 122),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_SUBTEXT, 1, cv2.LINE_AA)

    # Mini spinner inside the side panel card
    sp_cx = px + PW // 2
    sp_cy = 155
    sp_angle = int(time.time() * 300) % 360
    for i in range(8):
        a = sp_angle + i * 45
        rad = np.radians(a)
        sx = int(sp_cx + 18 * np.cos(rad))
        sy = int(sp_cy + 18 * np.sin(rad))
        alpha_c = max(40, 255 - i * 28)
        col = tuple(int(c * alpha_c / 255) for c in COLOR_ACCENT)
        cv2.circle(combined_display, (sx, sy), 4 - i // 3, col, -1)

    cv2.imshow("Driver Monitoring System", combined_display)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        sys.exit(0)

# ── Session Start Chime ───────────────────────────────────────────────────────
def play_chime(freq=880, duration_ms=180, volume=0.35):
    """Play a short synthesized chime using pygame."""
    try:
        import pygame
        sample_rate = 44100
        n_samples = int(sample_rate * duration_ms / 1000)
        buf = np.zeros((n_samples, 2), dtype=np.int16)
        for i in range(n_samples):
            t = i / sample_rate
            fade = 1.0 - (i / n_samples)
            val = int(32767 * volume * fade * np.sin(2 * np.pi * freq * t))
            buf[i] = [val, val]
        sound = pygame.mixer.Sound(buffer=buf)
        sound.play()
    except Exception:
        pass

play_chime(880, 180)          # High chime
time.sleep(0.12)
play_chime(1174, 220)         # Even higher follow-up
speak("DriveGuard is now active. Drive safe.")
session_start_time = time.time()   # Reset session clock precisely at monitoring start

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Save video frame if recording is active
    if recording:
        rec_frame = frame.copy()
        cv2.putText(rec_frame, "DROWSY EVENT", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        
        rec_timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(rec_frame, rec_timestamp, (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        video_writer.write(rec_frame)

    results = face_mesh.process(rgb)
    current_time = time.time()

    # Default keep previous state (do NOT reset blindly)
    new_state = state
    new_color = state_color

    if results.multi_face_landmarks:
        face_missing_start = None # Reset timer on face found
        for face_landmarks in results.multi_face_landmarks:

            # Extract landmarks for features
            left_eye, right_eye = [], []

            for idx in LEFT_EYE:
                x = int(face_landmarks.landmark[idx].x * w)
                y = int(face_landmarks.landmark[idx].y * h)
                left_eye.append((x, y))

            for idx in RIGHT_EYE:
                x = int(face_landmarks.landmark[idx].x * w)
                y = int(face_landmarks.landmark[idx].y * h)
                right_eye.append((x, y))

            mouth = []
            for idx in [13, 14, 78, 308]:
                x = int(face_landmarks.landmark[idx].x * w)
                y = int(face_landmarks.landmark[idx].y * h)
                mouth.append((x, y))

            # Calculate Ratios
            ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0
            mar = mouth_aspect_ratio(mouth)

            # Check for yawns
            if mar > MAR_THRESHOLD:
                if yawn_start_time is None:
                    yawn_start_time = current_time
                else:
                    if current_time - yawn_start_time >= YAWN_TIME:
                        yawn_count += 1
                        yawn_event_trigger = True
                        yawn_start_time = None
                        new_state = "SLEEPY"
                        new_color = (0, 255, 255)
                        sleepy_start_time = current_time
            else:
                yawn_start_time = None
                yawn_event_trigger = False

            # Check for closed eyes
            if ear < EAR_THRESHOLD:
                if eye_closed_start is None:
                    eye_closed_start = current_time
                else:
                    elapsed = current_time - eye_closed_start

                    if elapsed >= EMERGENCY_TIME:
                        # Send first email or repeat every 30 seconds
                        if last_emergency_email_time is None or (current_time - last_emergency_email_time) >= EMERGENCY_REPEAT_INTERVAL:
                            threading.Thread(target=send_emergency_email, daemon=True).start()
                            last_emergency_email_time = current_time
                            email_toast_time = current_time
                            speak("Emergency email sent. Please pull over safely.")

                    if elapsed >= DROWSY_TIME:
                        new_state = "DROWSY"
                        new_color = (0, 0, 255)
                        drowsy_active = True

                        if not alarm_on:
                            alarm_on = True
                            alarm_sound.play(-1)
                        
                        # Trigger Video Recording
                        if not recording:
                            timestamp = time.strftime("%Y%m%d_%H%M%S")
                            video_path = os.path.join(session_evidence_dir, f"drowsy_{timestamp}.mp4")
                            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                            video_writer = cv2.VideoWriter(video_path, fourcc, 20.0, (w, h))
                            recording = True
                            record_start_time = current_time

                    elif elapsed >= SLEEPY_BLINK_TIME:
                        new_state = "SLEEPY"
                        new_color = (0, 255, 255)

            else:
                if eye_closed_start is not None:
                    blink_times.append(current_time)

                    if state == "DROWSY":
                        new_state = "ACTIVE"
                        new_color = (0, 255, 0)
                        drowsy_active = False
                        # Only reset emergency timer on full DROWSY→ACTIVE recovery
                        last_emergency_email_time = None
                    
                    # Stop recording if recovered early
                    if recording:
                        video_writer.release()
                        recording = False

                    eye_closed_start = None

                if alarm_on:
                    alarm_on = False
                    alarm_sound.stop()

            # Solve Head Pose (PnP)
            face_2d = []
            face_3d = []

            for idx in POSE_LANDMARKS:
                x = int(face_landmarks.landmark[idx].x * w)
                y = int(face_landmarks.landmark[idx].y * h)

                face_2d.append([x, y])
                face_3d.append([x, y, face_landmarks.landmark[idx].z * 3000])

            face_2d = np.array(face_2d, dtype=np.float64)
            face_3d = np.array(face_3d, dtype=np.float64)

            nose_x = face_landmarks.landmark[NOSE_TIP].x
            left_cheek_x = face_landmarks.landmark[LEFT_CHEEK].x
            right_cheek_x = face_landmarks.landmark[RIGHT_CHEEK].x

            face_center_x = (left_cheek_x + right_cheek_x) / 2
            nose_offset = nose_x - face_center_x


            # Camera matrix
            focal_length = w
            cam_matrix = np.array([
                [focal_length, 0, w / 2],
                [0, focal_length, h / 2],
                [0, 0, 1]
            ])

            dist_matrix = np.zeros((4, 1), dtype=np.float64)

            success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

            rmat, _ = cv2.Rodrigues(rot_vec)
            angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

            pitch = angles[0]
            yaw   = angles[1] 
            roll  = angles[2]

            # Update Blink Rate (Sliding Window)
            while blink_times and current_time - blink_times[0] > 60:
                blink_times.popleft()

            blink_rate = len(blink_times)

            # Check Sleepy (Frequent Blinking)
            if blink_rate >= BLINK_RATE_THRESHOLD and new_state == "ACTIVE":
                new_state = "SLEEPY"
                new_color = (0, 255, 255)
                sleepy_start_time = current_time

            # Recovery Logic
            if state == "SLEEPY":
                if blink_rate < BLINK_RATE_THRESHOLD:
                    if sleepy_start_time and current_time - sleepy_start_time >= RECOVERY_TIME:
                        new_state = "ACTIVE"
                        new_color = (0, 255, 0)
                        sleepy_start_time = None

            # Distraction Detection
            looking_away = False

            if nose_offset > 0.035:
                looking_away = True
                direction = "RIGHT"
            elif nose_offset < -0.035:
                looking_away = True
                direction = "LEFT"
            elif pitch > 7:
                looking_away = True
                direction = "DOWN"
            else:
                direction = "FORWARD"

            if looking_away:
                distract_recovery_start = None
                if distraction_start_time is None:
                    distraction_start_time = current_time
                else:
                    if current_time - distraction_start_time >= DISTRACT_TIME:
                        # Mark as distracted if driver looks away for too long
                        if new_state == "ACTIVE":
                            new_state = "DISTRACTED"
                            new_color = (255, 0, 255)
            else:
                distraction_start_time = None
                if state == "DISTRACTED":
                    if distract_recovery_start is None:
                        distract_recovery_start = current_time
                    else:
                        if current_time - distract_recovery_start >= DISTRACT_RECOVERY_TIME:
                            new_state = "ACTIVE"
                            new_color = (0, 255, 0)
                            distract_recovery_start = None
            # Ensure Sleepy/Drowsy states persist over Distraction
            if state == "SLEEPY" and new_state == "DISTRACTED":
                new_state = "SLEEPY"
                new_color = (0, 255, 255)

            # Update Fatigue Score
            now = time.time()
            dt = now - last_fatigue_update
            last_fatigue_update = now

            fatigue_score = update_fatigue(
                fatigue_score,
                new_state,
                blink_rate,
                yawn_event_trigger,
                new_state == "DISTRACTED",
                drowsy_active,
                dt
            )
            # Reset event trigger
            if yawn_event_trigger:
                yawn_event_trigger = False

            # Update Level
            if fatigue_score < 30:
                fatigue_level = "SAFE"
            elif fatigue_score < 60:
                fatigue_level = "CAUTION"
            elif fatigue_score < 80:
                fatigue_level = "WARNING"
            else:
                fatigue_level = "CRITICAL"

            # Apply State
            state = new_state
            state_color = new_color

            # Log data to CSV every 1 second
            if current_time - last_csv_log_time >= 1.0:
                last_csv_log_time = current_time
                now = time.localtime()
                date_str = time.strftime("%Y-%m-%d", now)
                time_str = time.strftime("%H:%M:%S", now)

                with open(LOG_FILE, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([
                        date_str,
                        time_str,
                        state,
                        int(fatigue_score),
                        round(ear, 3),
                        blink_rate,
                        yawn_count,
                        direction,
                        round(nose_offset, 4),
                        round(mar, 3)
                    ])

                last_logged_state = state

            # ── Render Dashboard ───────────────────────────────────────────────
            combined_display = np.zeros((h, w + PANEL_WIDTH, 3), dtype=np.uint8)
            combined_display[:] = COLOR_BG
            combined_display[:h, :w] = frame

            PW = PANEL_WIDTH
            px = w  # panel left edge

            # Panel background
            combined_display[:h, px:px+PW] = COLOR_PANEL

            # ── HEADER STRIP ──────────────────────────────────────────────────
            cv2.rectangle(combined_display, (px, 0), (px + PW, 62), COLOR_PANEL2, -1)
            cv2.rectangle(combined_display, (px, 60), (px + PW, 62), COLOR_ACCENT, -1)

            # DG logo block (solid accent box on far left)
            cv2.rectangle(combined_display, (px, 0), (px + 58, 62), COLOR_ACCENT, -1)
            cv2.putText(combined_display, "DG", (px + 8, 42),
                        cv2.FONT_HERSHEY_DUPLEX, 0.95, (10, 15, 25), 2, cv2.LINE_AA)

            # Title and subtitle
            cv2.putText(combined_display, "DriveGuard", (px + 68, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.56, COLOR_TEXT, 1, cv2.LINE_AA)
            cv2.putText(combined_display, "Driver Monitoring System", (px + 68, 46),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, COLOR_SUBTEXT, 1, cv2.LINE_AA)

            # Live clock + date - right-aligned
            live_date = time.strftime("%d %b %Y")
            live_time = time.strftime("%H:%M:%S")

            ts_w = cv2.getTextSize(live_time, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)[0][0]
            dt_w = cv2.getTextSize(live_date, cv2.FONT_HERSHEY_SIMPLEX, 0.34, 1)[0][0]
            cv2.putText(combined_display, live_time,
                        (px + PW - ts_w - 10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, COLOR_ACCENT, 1, cv2.LINE_AA)
            cv2.putText(combined_display, live_date,
                        (px + PW - dt_w - 10, 48),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.34, COLOR_SUBTEXT, 1, cv2.LINE_AA)

            py = 78
            CARD_X1 = px + 12
            CARD_X2 = px + PW - 12

            # ── STATUS CARD ──────────────────────────────────────────────────
            # State → fill color mapping  (BGR)
            status_map = {
                "ACTIVE":     (55,  200,  90),   # green
                "SLEEPY":     (40,  210, 210),   # yellow-amber
                "DROWSY":     (45,   55, 220),   # red
                "DISTRACTED": (190,  60, 200),   # purple
            }
            s_col = status_map.get(state, COLOR_ACCENT)

            # Fully filled rounded card in state color
            draw_rounded_rect(combined_display, CARD_X1, py, CARD_X2, py + 72, 8, s_col)

            # Choose contrasting text color: dark on bright states, white on dark
            bright_states = {"ACTIVE", "SLEEPY"}
            txt_col = (15, 20, 30) if state in bright_states else (240, 245, 255)
            sub_col = (30, 40, 50) if state in bright_states else (190, 200, 220)

            # Labels
            cv2.putText(combined_display, "DRIVER STATUS", (CARD_X1 + 16, py + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, sub_col, 1, cv2.LINE_AA)
            cv2.putText(combined_display, state, (CARD_X1 + 16, py + 58),
                        cv2.FONT_HERSHEY_DUPLEX, 1.15, txt_col, 2, cv2.LINE_AA)

            # Solid indicator dot (contrasting)
            dot_x = CARD_X2 - 28
            cv2.circle(combined_display, (dot_x, py + 36), 10, txt_col, -1)

            py += 86

            # ── EYE TRACKING CARD ─────────────────────────────────────────────
            draw_rounded_rect(combined_display, CARD_X1, py, CARD_X2, py + 100, 8, COLOR_PANEL2)
            draw_section_header(combined_display, CARD_X1 + 12, py + 20, "EYE TRACKING", COLOR_ACCENT)

            # EAR bar
            ear_color = COLOR_OK if ear > EAR_THRESHOLD else COLOR_RED
            cv2.putText(combined_display, "EAR", (CARD_X1 + 12, py + 48),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_SUBTEXT, 1, cv2.LINE_AA)
            bar_w = CARD_X2 - CARD_X1 - 80
            draw_bar(combined_display, CARD_X1 + 40, py + 36, bar_w, 14, ear, 0.4, ear_color)
            cv2.putText(combined_display, f"{ear:.2f}", (CARD_X1 + 44 + bar_w, py + 48),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, ear_color, 1, cv2.LINE_AA)

            # Blink & Yawn chips
            cx = CARD_X1 + 12
            cy = py + 74
            # Blink chip
            chip_w = 120
            draw_rounded_rect(combined_display, cx, cy - 16, cx + chip_w, cy + 8, 6, COLOR_PANEL)
            cv2.putText(combined_display, f"Blinks  {blink_rate}/min", (cx + 8, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_ACCENT, 1, cv2.LINE_AA)
            # Yawn chip
            cx2 = cx + chip_w + 10
            draw_rounded_rect(combined_display, cx2, cy - 16, cx2 + 100, cy + 8, 6, COLOR_PANEL)
            cv2.putText(combined_display, f"Yawns   {yawn_count}", (cx2 + 8, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_WARN, 1, cv2.LINE_AA)

            py += 114

            # ── HEAD POSTURE CARD ─────────────────────────────────────────────
            draw_rounded_rect(combined_display, CARD_X1, py, CARD_X2, py + 108, 8, COLOR_PANEL2)
            draw_section_header(combined_display, CARD_X1 + 12, py + 20, "HEAD POSTURE", COLOR_ACCENT)

            dir_colors = {
                "FORWARD": COLOR_OK,
                "LEFT":    COLOR_PURP,
                "RIGHT":   COLOR_PURP,
                "DOWN":    COLOR_RED,
            }
            d_col = dir_colors.get(direction, COLOR_TEXT)

            # Direction badge
            badge_text = direction
            bt_w = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0][0]
            draw_pill(combined_display,
                      CARD_X1 + 12, py + 30,
                      CARD_X1 + 20 + bt_w + 16, py + 56, d_col)
            cv2.putText(combined_display, badge_text, (CARD_X1 + 20, py + 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (10, 15, 25), 2, cv2.LINE_AA)

            # Angle data
            cv2.putText(combined_display,
                        f"Nose offset: {nose_offset:+.3f}   Yaw: {int(yaw):+d}   Pitch: {int(pitch):+d}",
                        (CARD_X1 + 12, py + 88),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_SUBTEXT, 1, cv2.LINE_AA)

            py += 122

            # ── FATIGUE SCORE CARD ────────────────────────────────────────────
            draw_rounded_rect(combined_display, CARD_X1, py, CARD_X2, py + 100, 8, COLOR_PANEL2)
            draw_section_header(combined_display, CARD_X1 + 12, py + 20, "FATIGUE INDEX", COLOR_ACCENT)

            bar_color = (COLOR_OK if fatigue_score < 30
                         else COLOR_WARN if fatigue_score < 60
                         else COLOR_RED)
            full_bar_w = CARD_X2 - CARD_X1 - 24
            draw_bar(combined_display, CARD_X1 + 12, py + 32, full_bar_w, 16,
                     fatigue_score, 100, bar_color)

            # Score number large
            score_str = f"{int(fatigue_score)}%"
            cv2.putText(combined_display, score_str, (CARD_X1 + 12, py + 82),
                        cv2.FONT_HERSHEY_DUPLEX, 1.0, bar_color, 2, cv2.LINE_AA)
            # Level label pill
            lv_x = CARD_X1 + 12 + cv2.getTextSize(score_str, cv2.FONT_HERSHEY_DUPLEX, 1.0, 2)[0][0] + 14
            lv_w = cv2.getTextSize(fatigue_level, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0][0] + 20
            draw_pill(combined_display, lv_x, py + 64, lv_x + lv_w, py + 86, bar_color)
            cv2.putText(combined_display, fatigue_level, (lv_x + 10, py + 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (10, 15, 25), 1, cv2.LINE_AA)

            py += 112

            # ── SESSION TIMER CARD ────────────────────────────────────────────
            elapsed = time.time() - session_start_time
            el_h = int(elapsed // 3600)
            el_m = int((elapsed % 3600) // 60)
            el_s = int(elapsed % 60)
            timer_str = f"{el_h:02d}:{el_m:02d}:{el_s:02d}"

            draw_rounded_rect(combined_display, CARD_X1, py, CARD_X2, py + 100, 8, COLOR_PANEL2)
            draw_section_header(combined_display, CARD_X1 + 12, py + 20, "SESSION TIMER", COLOR_ACCENT)

            cv2.putText(combined_display, timer_str, (CARD_X1 + 12, py + 46),
                        cv2.FONT_HERSHEY_DUPLEX, 0.72, COLOR_TEXT, 1, cv2.LINE_AA)

            # ── Break progress bar (dedicated row) ────────────────────────────
            if BREAK_REMINDER_SEC > 0:
                break_progress = min(1.0, elapsed / BREAK_REMINDER_SEC)
                bp_color = COLOR_OK if break_progress < 0.75 else (COLOR_WARN if break_progress < 1.0 else COLOR_RED)

                # "NEXT BREAK" label
                cv2.putText(combined_display, "NEXT BREAK", (CARD_X1 + 12, py + 66),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_SUBTEXT, 1, cv2.LINE_AA)

                # Full-width progress bar
                bp_bar_w = CARD_X2 - CARD_X1 - 24
                draw_bar(combined_display, CARD_X1 + 12, py + 72, bp_bar_w, 14,
                         break_progress, 1.0, bp_color)

                # Remaining time label (right-aligned)
                remaining_sec = max(0, BREAK_REMINDER_SEC - elapsed)
                rem_m = int(remaining_sec // 60)
                rem_s = int(remaining_sec % 60)
                rem_str = f"{rem_m}m {rem_s}s" if remaining_sec > 0 else "BREAK TIME!"
                rem_w = cv2.getTextSize(rem_str, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)[0][0]
                cv2.putText(combined_display, rem_str,
                            (CARD_X2 - rem_w - 12, py + 66),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, bp_color, 1, cv2.LINE_AA)

            py += 112

            # ── BREAK REMINDER LOGIC ──────────────────────────────────────────
            if BREAK_REMINDER_SEC > 0 and elapsed >= BREAK_REMINDER_SEC:
                now_t = time.time()
                # Trigger voice reminder every BREAK_REMINDER_SEC interval
                if now_t - last_break_reminder_time >= BREAK_REMINDER_SEC:
                    last_break_reminder_time = now_t
                    break_reminder_shown = True
                    break_banner_time = now_t
                    speak("You have been driving for a long time. Please take a break and rest for a while.")

            # Show break banner for 10 seconds after trigger
            if break_reminder_shown and break_banner_time is not None:
                if time.time() - break_banner_time < 10.0:
                    blink_on = int(time.time() * 2) % 2 == 0
                    banner_col = COLOR_ACCENT if blink_on else COLOR_WARN
                    draw_rounded_rect(combined_display, CARD_X1, py, CARD_X2, py + 32, 8, COLOR_PANEL2)
                    cv2.putText(combined_display, "Take a break! Rest for safety.",
                                (CARD_X1 + 12, py + 22),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.44, banner_col, 1, cv2.LINE_AA)
                    py += 40
                else:
                    break_reminder_shown = False

            # ── RECORDING BADGE ───────────────────────────────────────────────
            if recording:
                rec_blink = int(time.time() * 2) % 2 == 0
                if rec_blink:
                    draw_pill(combined_display,
                              CARD_X1, py, CARD_X1 + 130, py + 28, COLOR_RED)
                    cv2.circle(combined_display, (CARD_X1 + 20, py + 14), 7, (10,15,25), -1)
                    cv2.putText(combined_display, "REC", (CARD_X1 + 32, py + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (10, 15, 25), 2, cv2.LINE_AA)

            # ── EMAIL TOAST ───────────────────────────────────────────────────
            if email_toast_time is not None and (time.time() - email_toast_time) < 2.0:
                # Side-panel pill
                toast_color = (60, 180, 80)
                draw_rounded_rect(combined_display, CARD_X1, py + 36, CARD_X2, py + 64, 8, toast_color)
                cv2.putText(combined_display, "EMAIL  sent!",
                            (CARD_X1 + 14, py + 58),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)

                # Camera-feed centred popup banner
                banner_h = 72
                banner_y1 = h // 2 - banner_h // 2
                banner_y2 = banner_y1 + banner_h
                blend_rect(combined_display, 0, banner_y1, w, banner_y2, (30, 120, 50), alpha=0.82)
                cv2.line(combined_display, (0, banner_y1), (w, banner_y1), (80, 220, 130), 2)
                cv2.line(combined_display, (0, banner_y2), (w, banner_y2), (80, 220, 130), 2)
                line1 = "Emergency Email Sent!"
                line2 = "Help is on the way"
                l1_w = cv2.getTextSize(line1, cv2.FONT_HERSHEY_DUPLEX, 0.85, 2)[0][0]
                l2_w = cv2.getTextSize(line2, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)[0][0]
                cv2.putText(combined_display, line1,
                            ((w - l1_w) // 2, banner_y1 + 40),
                            cv2.FONT_HERSHEY_DUPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(combined_display, line2,
                            ((w - l2_w) // 2, banner_y1 + 62),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180, 240, 200), 1, cv2.LINE_AA)

            # ── FOOTER ────────────────────────────────────────────────────────
            fy = h - 20
            cv2.rectangle(combined_display, (px, fy - 14), (px + PW, h), COLOR_PANEL2, -1)
            cv2.putText(combined_display, "Press  Q  to quit",
                        (px + 14, fy), cv2.FONT_HERSHEY_SIMPLEX,
                        0.38, COLOR_SUBTEXT, 1, cv2.LINE_AA)

            # ── CAMERA BORDER ─────────────────────────────────────────────────
            # Thin coloured border around camera pane matching state
            cv2.rectangle(combined_display, (0, 0), (w - 1, h - 1), s_col, 2)

            cv2.imshow("Driver Monitoring System", combined_display)

    else:
        # ── Face Not Detected ─────────────────────────────────────────────────
        if face_missing_start is None:
            face_missing_start = time.time()

        elapsed_missing = time.time() - face_missing_start

        combined_display = np.zeros((h, w + PANEL_WIDTH, 3), dtype=np.uint8)
        combined_display[:] = COLOR_BG
        combined_display[:h, :w] = frame

        # Panel bg
        PW = PANEL_WIDTH
        combined_display[:h, w:w+PW] = COLOR_PANEL
        cv2.rectangle(combined_display, (w, 0), (w + PW, 62), COLOR_PANEL2, -1)
        cv2.rectangle(combined_display, (w, 60), (w + PW, 62), COLOR_ACCENT, -1)
        cv2.rectangle(combined_display, (w, 0), (w + 58, 62), COLOR_ACCENT, -1)
        cv2.putText(combined_display, "DG", (w + 8, 40),
                    cv2.FONT_HERSHEY_DUPLEX, 0.95, (10, 15, 25), 2, cv2.LINE_AA)
        cv2.putText(combined_display, "DriveGuard", (w + 68, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.56, COLOR_TEXT, 1, cv2.LINE_AA)
        cv2.putText(combined_display, "Driver Monitoring System", (w + 68, 46),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, COLOR_SUBTEXT, 1, cv2.LINE_AA)

        # Use longer threshold if last state was DISTRACTED (head-turn loses face)
        effective_threshold = (NO_FACE_DISTRACTED_THRESHOLD
                               if state == "DISTRACTED"
                               else NO_FACE_THRESHOLD)

        if elapsed_missing >= effective_threshold:
            # Pulsing red vignette overlay on camera
            pulse = 0.25 + 0.12 * abs(np.sin(time.time() * 3))
            blend_rect(combined_display, 0, 0, w, h, (30, 30, 200), alpha=pulse)

            # Warning banner at bottom of camera
            blend_rect(combined_display, 0, h - 60, w, h, (0, 0, 0), alpha=0.6)
            warn_text = "FACE NOT DETECTED"
            wt_size = cv2.getTextSize(warn_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
            wt_x = (w - wt_size[0]) // 2
            cv2.putText(combined_display, warn_text, (wt_x, h - 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

            # Panel status
            draw_rounded_rect(combined_display, w + 12, 78, w + PW - 12, 148, 8, COLOR_PANEL2)
            draw_rounded_rect(combined_display, w + 12, 78, w + PW - 12, 148, 8, COLOR_RED, thickness=1)
            draw_rounded_rect(combined_display, w + 12, 78, w + 17, 148, 2, COLOR_RED)
            cv2.putText(combined_display, "DRIVER STATUS", (w + 28, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, COLOR_SUBTEXT, 1, cv2.LINE_AA)
            cv2.putText(combined_display, "SEARCHING", (w + 28, 136),
                        cv2.FONT_HERSHEY_DUPLEX, 0.9, COLOR_RED, 2, cv2.LINE_AA)

            # Red border on camera pane
            cv2.rectangle(combined_display, (0, 0), (w - 1, h - 1), COLOR_RED, 2)
        else:
            # Buffer period — neutral
            draw_rounded_rect(combined_display, w + 12, 78, w + PW - 12, 148, 8, COLOR_PANEL2)
            cv2.putText(combined_display, "DRIVER STATUS", (w + 28, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, COLOR_SUBTEXT, 1, cv2.LINE_AA)
            cv2.putText(combined_display, "WAIT", (w + 28, 136),
                        cv2.FONT_HERSHEY_DUPLEX, 0.9, COLOR_SUBTEXT, 2, cv2.LINE_AA)

        # Footer
        fy = h - 20
        cv2.rectangle(combined_display, (w, fy - 14), (w + PW, h), COLOR_PANEL2, -1)
        cv2.putText(combined_display, "Press  Q  to quit",
                    (w + 14, fy), cv2.FONT_HERSHEY_SIMPLEX,
                    0.38, COLOR_SUBTEXT, 1, cv2.LINE_AA)

        cv2.imshow("Driver Monitoring System", combined_display)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ── Session End Chime ─────────────────────────────────────────────────────────
play_chime(1174, 180)         # High note
time.sleep(0.12)
play_chime(880, 220)          # Descending follow-up
speak("Session ended. Thank you for using DriveGuard.")
time.sleep(0.3)

# ── Cleanup ──────────────────────────────────────────────────────────────────
if alarm_on and alarm_sound is not None:
    try:
        alarm_sound.stop()
    except Exception:
        pass

if recording and video_writer is not None:
    try:
        video_writer.release()
    except Exception:
        pass

cap.release()
cv2.destroyAllWindows()

# ══════════════════════════════════════════════════════════════════════════════
#  POST-SESSION: REPORT GENERATION PROMPT
# ══════════════════════════════════════════════════════════════════════════════

def launch_post_session_ui():
    """
    Show styled Tkinter windows after monitoring ends:
    1. Ask if user wants an analytical report  (Y / N)
    2. If Y → file-picker to select / confirm CSV path
    3. Run analysis → generate PDF → auto-open it
    """
    import tkinter as tk
    from tkinter import filedialog

    # Try to import tkinterdnd2 for drag-and-drop support
    try:
        from tkinterdnd2 import TkinterDnD, DND_FILES
        HAS_DND = True
    except ImportError:
        HAS_DND = False

    # ── Colour constants (matching DriveGuard dashboard) ─────────────────────
    BG            = "#121624"
    PANEL         = "#1C2134"
    PANEL2        = "#242840"
    BORDER_CLR    = "#37415F"
    TEXT_CLR      = "#DCE4F5"
    SUBTEXT_CLR   = "#7887AA"
    ACCENT_CLR    = "#50C8FF"
    ACCENT_HOVER  = "#6DD4FF"
    OK_CLR        = "#3CDC8C"
    DANGER_CLR    = "#F05050"
    BTN_GRAY      = "#2A3248"
    BTN_GRAY_HVR  = "#3A4560"

    # ── Helper: centre a window on screen ────────────────────────────────────
    def centre_window(win, w, h):
        sw = win.winfo_screenwidth()
        sh = win.winfo_screenheight()
        win.geometry(f"{w}x{h}+{(sw - w) // 2}+{(sh - h) // 2}")

    # ══════════════════════════════════════════════════════════════════════════
    #  WINDOW 1 — "Generate Report?"
    # ══════════════════════════════════════════════════════════════════════════

    def show_report_prompt():
        root = TkinterDnD.Tk() if HAS_DND else tk.Tk()
        root.title("DriveGuard \u2014 Session Complete")
        root.configure(bg=BG)
        root.resizable(False, False)
        centre_window(root, 480, 260)

        # Prevent closing via X from leaving a zombie — treat as "No"
        root.protocol("WM_DELETE_WINDOW", lambda: (setattr(choice, 'val', False), root.destroy()))

        # ── Header bar ───────────────────────────────────────────────────────
        hdr = tk.Frame(root, bg=PANEL2, height=50)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)

        tk.Label(hdr, text="  DG  ", bg=ACCENT_CLR, fg="#0A0F19",
                 font=("Segoe UI", 14, "bold")).pack(side="left", fill="y")
        tk.Label(hdr, text="DriveGuard", bg=PANEL2, fg=TEXT_CLR,
                 font=("Segoe UI", 13, "bold")).pack(side="left", padx=(10, 0))
        tk.Label(hdr, text="Session Complete", bg=PANEL2, fg=SUBTEXT_CLR,
                 font=("Segoe UI", 9)).pack(side="left", padx=(8, 0))

        tk.Frame(root, bg=ACCENT_CLR, height=2).pack(fill="x")

        # -- Body -------------------------------------------------------------
        body = tk.Frame(root, bg=BG)
        body.pack(fill="both", expand=True, padx=30, pady=18)

        tk.Label(body, text="\u2713", bg=BG, fg=OK_CLR,
                 font=("Segoe UI", 30, "bold")).pack()
        tk.Label(body,
                 text="Session monitoring complete.\nWould you like to generate an analytical report?",
                 bg=BG, fg=TEXT_CLR, font=("Segoe UI", 11),
                 justify="center").pack(pady=(4, 22))

        # Choice container
        class choice:
            val = False

        bf = tk.Frame(body, bg=BG)
        bf.pack()

        def on_yes():
            choice.val = True
            root.destroy()

        yes_btn = tk.Button(bf, text="  Yes, Generate Report  ",
                            bg=ACCENT_CLR, fg="#0A0F19",
                            font=("Segoe UI", 11, "bold"), relief="flat",
                            cursor="hand2", activebackground=ACCENT_HOVER,
                            activeforeground="#0A0F19", command=on_yes,
                            bd=0, padx=20, pady=8)
        yes_btn.pack(side="left", padx=(0, 14))

        no_btn = tk.Button(bf, text="  No, Exit  ",
                           bg=BTN_GRAY, fg=SUBTEXT_CLR,
                           font=("Segoe UI", 11), relief="flat",
                           cursor="hand2", activebackground=BTN_GRAY_HVR,
                           activeforeground=TEXT_CLR,
                           command=root.destroy, bd=0, padx=20, pady=8)
        no_btn.pack(side="left")

        # Hover effects
        yes_btn.bind("<Enter>", lambda e: yes_btn.config(bg=ACCENT_HOVER))
        yes_btn.bind("<Leave>", lambda e: yes_btn.config(bg=ACCENT_CLR))
        no_btn.bind("<Enter>", lambda e: no_btn.config(bg=BTN_GRAY_HVR, fg=TEXT_CLR))
        no_btn.bind("<Leave>", lambda e: no_btn.config(bg=BTN_GRAY, fg=SUBTEXT_CLR))

        root.mainloop()
        return choice.val

    # --------------------------------------------------------------------------
    #  WINDOW 2 - File Picker + Analysis Runner
    # --------------------------------------------------------------------------

    def show_file_picker():
        root = TkinterDnD.Tk() if HAS_DND else tk.Tk()
        root.title("DriveGuard \u2014 Select Session File")
        root.configure(bg=BG)
        root.resizable(False, False)
        centre_window(root, 600, 400)
        root.protocol("WM_DELETE_WINDOW", root.destroy)

        # -- Header -----------------------------------------------------------
        hdr = tk.Frame(root, bg=PANEL2, height=50)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)

        tk.Label(hdr, text="  DG  ", bg=ACCENT_CLR, fg="#0A0F19",
                 font=("Segoe UI", 14, "bold")).pack(side="left", fill="y")
        tk.Label(hdr, text="DriveGuard", bg=PANEL2, fg=TEXT_CLR,
                 font=("Segoe UI", 13, "bold")).pack(side="left", padx=(10, 0))
        tk.Label(hdr, text="Session Analysis", bg=PANEL2, fg=SUBTEXT_CLR,
                 font=("Segoe UI", 9)).pack(side="left", padx=(8, 0))

        tk.Frame(root, bg=ACCENT_CLR, height=2).pack(fill="x")

        # -- Body -------------------------------------------------------------
        body = tk.Frame(root, bg=BG)
        body.pack(fill="both", expand=True, padx=28, pady=16)

        tk.Label(body, text="Select the session CSV file to analyze",
                 bg=BG, fg=TEXT_CLR,
                 font=("Segoe UI", 12, "bold")).pack(anchor="w")
        tk.Label(body,
                 text="Paste a path, browse for a file, or drag & drop a CSV below",
                 bg=BG, fg=SUBTEXT_CLR,
                 font=("Segoe UI", 9)).pack(anchor="w", pady=(2, 14))

        # -- Drop zone card ---------------------------------------------------
        drop = tk.Frame(body, bg=PANEL2, highlightbackground=BORDER_CLR,
                        highlightthickness=2, padx=16, pady=14)
        drop.pack(fill="x")

        tk.Label(drop, text="CSV File Path:", bg=PANEL2, fg=SUBTEXT_CLR,
                 font=("Segoe UI", 9)).pack(anchor="w")

        path_var = tk.StringVar(value=os.path.abspath(LOG_FILE))

        entry = tk.Entry(drop, textvariable=path_var, bg="#0E1220", fg=TEXT_CLR,
                         font=("Consolas", 10), insertbackground=ACCENT_CLR,
                         relief="flat", highlightbackground=BORDER_CLR,
                         highlightthickness=1, highlightcolor=ACCENT_CLR)
        entry.pack(fill="x", pady=(4, 10), ipady=6)

        # Browse button
        def browse():
            fp = filedialog.askopenfilename(
                title="Select Session CSV",
                initialdir=os.path.abspath(RECORDS_DIR),
                filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
            )
            if fp:
                path_var.set(fp)

        browse_btn = tk.Button(drop, text="  \U0001f4c1  Browse Files  ",
                               bg=BTN_GRAY, fg=TEXT_CLR,
                               font=("Segoe UI", 10), relief="flat",
                               cursor="hand2", activebackground=BTN_GRAY_HVR,
                               command=browse, bd=0, padx=12, pady=4)
        browse_btn.pack(anchor="w")
        browse_btn.bind("<Enter>", lambda e: browse_btn.config(bg=BTN_GRAY_HVR))
        browse_btn.bind("<Leave>", lambda e: browse_btn.config(bg=BTN_GRAY))

        # Drag-and-drop zone
        if HAS_DND:
            dnd_label = tk.Label(drop,
                                 text="\u2014 or drag & drop a CSV file here \u2014",
                                 bg=PANEL2, fg=SUBTEXT_CLR,
                                 font=("Segoe UI", 9, "italic"))
            dnd_label.pack(pady=(10, 0))

            def on_drop(event):
                path = event.data.strip()
                # Windows wraps multi-word paths in {}
                if path.startswith("{") and path.endswith("}"):
                    path = path[1:-1]
                path_var.set(path)
                drop.config(highlightbackground=OK_CLR)
                root.after(1200, lambda: drop.config(highlightbackground=BORDER_CLR))

            drop.drop_target_register(DND_FILES)
            drop.dnd_bind("<<Drop>>", on_drop)

        # -- Status label -----------------------------------------------------
        status_var = tk.StringVar(value="")
        status_lbl = tk.Label(body, textvariable=status_var, bg=BG,
                              fg=SUBTEXT_CLR, font=("Segoe UI", 9))
        status_lbl.pack(anchor="w", pady=(12, 0))

        # -- Action buttons ---------------------------------------------------
        bf = tk.Frame(body, bg=BG)
        bf.pack(pady=(14, 0))

        def on_analyze():
            csv_path = path_var.get().strip().strip('"').strip("'")

            if not csv_path or not os.path.isfile(csv_path):
                status_var.set("\u26a0  File not found. Please check the path.")
                status_lbl.config(fg=DANGER_CLR)
                return

            # Show progress
            status_var.set(">>  Analyzing session data...")
            status_lbl.config(fg=ACCENT_CLR)
            analyze_btn.config(state="disabled", bg=BTN_GRAY)
            cancel_btn.config(state="disabled")
            root.update()

            try:
                # Import analysis functions (same directory)
                from session_analysis import (
                    load_csv, compute_summary, generate_recommendations,
                    chart_state_timeline, chart_fatigue_over_time,
                    chart_ear_over_time, chart_blink_rate,
                    chart_head_direction_pie, chart_state_distribution,
                    generate_pdf_report, print_console_summary,
                    REPORTS_DIR as ANALYSIS_REPORTS_DIR,
                )

                df = load_csv(csv_path)
                summary = compute_summary(df)
                summary["recommendations"] = generate_recommendations(summary)
                print_console_summary(summary)

                os.makedirs(ANALYSIS_REPORTS_DIR, exist_ok=True)
                session_name = os.path.splitext(os.path.basename(csv_path))[0]
                report_folder = os.path.join(ANALYSIS_REPORTS_DIR,
                                             f"analysis_{session_name}")
                charts_folder = os.path.join(report_folder, "charts")
                os.makedirs(charts_folder, exist_ok=True)

                status_var.set(">>  Generating charts...")
                root.update()

                chart_funcs = {
                    "state_timeline":     chart_state_timeline,
                    "fatigue_over_time":  chart_fatigue_over_time,
                    "ear_over_time":      chart_ear_over_time,
                    "blink_rate":         chart_blink_rate,
                    "head_direction_pie": chart_head_direction_pie,
                    "state_distribution": chart_state_distribution,
                }
                chart_paths = {}
                for key, func in chart_funcs.items():
                    out = os.path.join(charts_folder, f"{key}.png")
                    try:
                        func(df, out)
                        chart_paths[key] = out
                    except Exception:
                        pass

                status_var.set(">>  Building PDF report...")
                root.update()

                pdf_path = os.path.join(report_folder,
                                        f"{session_name}_report.pdf")
                generate_pdf_report(df, summary, chart_paths, pdf_path)

                # Success
                status_var.set(f"\u2713  Report saved \u2192 {pdf_path}")
                status_lbl.config(fg=OK_CLR)
                analyze_btn.config(text="  \u2713  Done  ")
                root.update()

                # Auto-open PDF in default viewer (cross-platform)
                try:
                    abs_pdf = os.path.abspath(pdf_path)
                    if platform.system() == "Windows":
                        os.startfile(abs_pdf)
                    elif platform.system() == "Darwin":  # macOS
                        subprocess.run(["open", abs_pdf])
                    else:  # Linux
                        subprocess.run(["xdg-open", abs_pdf])
                except Exception:
                    pass

                root.after(2500, root.destroy)

            except Exception as exc:
                status_var.set(f"\u26a0  Error: {exc}")
                status_lbl.config(fg=DANGER_CLR)
                analyze_btn.config(state="normal", bg=ACCENT_CLR,
                                   text="  Analyze & Generate Report  ")
                cancel_btn.config(state="normal")

        analyze_btn = tk.Button(bf, text="  Analyze & Generate Report  ",
                                bg=ACCENT_CLR, fg="#0A0F19",
                                font=("Segoe UI", 11, "bold"), relief="flat",
                                cursor="hand2", activebackground=ACCENT_HOVER,
                                activeforeground="#0A0F19", command=on_analyze,
                                bd=0, padx=20, pady=8)
        analyze_btn.pack(side="left", padx=(0, 14))

        cancel_btn = tk.Button(bf, text="  Cancel  ",
                               bg=BTN_GRAY, fg=SUBTEXT_CLR,
                               font=("Segoe UI", 11), relief="flat",
                               cursor="hand2", activebackground=BTN_GRAY_HVR,
                               activeforeground=TEXT_CLR,
                               command=root.destroy, bd=0, padx=20, pady=8)
        cancel_btn.pack(side="left")

        # Hover effects
        def on_enter_a(e):
            if str(analyze_btn["state"]) != "disabled":
                analyze_btn.config(bg=ACCENT_HOVER)
        def on_leave_a(e):
            if str(analyze_btn["state"]) != "disabled":
                analyze_btn.config(bg=ACCENT_CLR)

        analyze_btn.bind("<Enter>", on_enter_a)
        analyze_btn.bind("<Leave>", on_leave_a)
        cancel_btn.bind("<Enter>", lambda e: cancel_btn.config(bg=BTN_GRAY_HVR, fg=TEXT_CLR))
        cancel_btn.bind("<Leave>", lambda e: cancel_btn.config(bg=BTN_GRAY, fg=SUBTEXT_CLR))

        root.mainloop()

    # ── Execute the flow ─────────────────────────────────────────────────────
    if show_report_prompt():
        show_file_picker()

launch_post_session_ui()
