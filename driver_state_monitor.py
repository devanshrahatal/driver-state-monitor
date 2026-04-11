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

# ── Premium UI Palette ────────────────────────────────────────────────────────
COLOR_BG       = (18,  22,  36)    # Deep navy background
COLOR_PANEL    = (28,  33,  52)    # Card / panel bg
COLOR_PANEL2   = (36,  42,  64)    # Slightly lighter card
COLOR_BORDER   = (55,  65,  95)    # Subtle border
COLOR_TEXT     = (220, 228, 245)   # Soft white text
COLOR_SUBTEXT  = (120, 135, 170)   # Dimmed subtext
COLOR_ACCENT   = (80,  200, 255)   # Cyan-blue accent
COLOR_WARN     = (45,  210, 165)   # Mint (sleepy)
COLOR_YEL      = (50,  205, 240)   # Info alternate
COLOR_DANGER   = (60,   80, 255)   # Electric red-blue (drowsy)
COLOR_RED      = (50,   80, 240)   # Pure alert red
COLOR_OK       = (60,  220, 140)   # Bright green (active)
COLOR_PURP     = (200,  80, 240)   # Purple (distracted)
COLOR_DARK_BAR = (35,  41,  62)    # Progress bar track

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

# Configuration Thresholds
EAR_THRESHOLD = 0.23
DROWSY_TIME = 2.0
SLEEPY_BLINK_TIME = 0.3
BLINK_RATE_THRESHOLD = 25
RECOVERY_TIME = 10
MAR_THRESHOLD = 0.6
YAWN_TIME = 2.0

DISTRACT_TIME = 2.0
DISTRACT_RECOVERY_TIME = 0.5
PANEL_WIDTH = 550

# Emergency Email Settings
EMERGENCY_TIME = 15.0
SENDER_EMAIL = "devanshrahatal@gmail.com"
SENDER_PASSWORD = "hlhq elvg fggz gium"
RECEIVER_EMAIL = "devanshrahatal@gmail.com"

EMERGENCY_REPEAT_INTERVAL = 30.0  # Resend email every 30 seconds while eyes are closed

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
NO_FACE_THRESHOLD = 1.5          # Seconds before showing "Face Not Detected"
NO_FACE_DISTRACTED_THRESHOLD = 3.0  # Extra grace period after a distraction head-turn

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
RECORD_DURATION = 5

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
                    last_emergency_email_time = None

                    if state == "DROWSY":
                        new_state = "ACTIVE"
                        new_color = (0, 255, 0)
                        drowsy_active = False
                    
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

            # Title and date — starts safely after the logo box
            cv2.putText(combined_display, "DrivrGuard", (px + 68, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.56, COLOR_TEXT, 1, cv2.LINE_AA)
            cv2.putText(combined_display, "Driver Monitoring System", (px + 68, 43),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, COLOR_SUBTEXT, 1, cv2.LINE_AA)
            live_date = time.strftime("%d %b %Y")
            cv2.putText(combined_display, live_date, (px + 68, 46),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_SUBTEXT, 1, cv2.LINE_AA)

            # Live clock — right-aligned
            live_time = time.strftime("%H:%M:%S")
            ts_w = cv2.getTextSize(live_time, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)[0][0]
            cv2.putText(combined_display, live_time,
                        (px + PW - ts_w - 10, 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, COLOR_ACCENT, 1, cv2.LINE_AA)

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
            draw_section_header(combined_display, CARD_X1 + 12, py + 20, "HEAD POSTURE", COLOR_YEL)

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
            draw_section_header(combined_display, CARD_X1 + 12, py + 20, "FATIGUE INDEX", COLOR_OK)

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

cap.release()
cv2.destroyAllWindows()
