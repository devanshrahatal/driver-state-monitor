"""
Driver Monitoring System
Features:
- Eye-based drowsiness detection
- Head pose–based distraction detection
- Fatigue score estimation
- Evidence video recording
- CSV-based event logging

"""

import os
import sys
import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance as dist
import time
import pygame
import csv
from collections import deque


# UI Colors
COLOR_BG = (30, 30, 30)       # Dark background
COLOR_PANEL = (50, 50, 50)    # Panel background
COLOR_TEXT = (220, 220, 220)  # Text color
COLOR_ACCENT = (0, 150, 255)  # Blue
COLOR_WARN = (0, 255, 255)    # Yellow
COLOR_DANGER = (0, 0, 255)    # Red
COLOR_OK = (0, 255, 0)        # Green

# Progress bar for stats
def draw_bar(image, x, y, w, h, value, max_value, color):
    cv2.rectangle(image, (x, y), (x + w, y + h), (70, 70, 70), -1)
    
    val_clamped = max(0, min(value, max_value))
    fill_w = int((val_clamped / max_value) * w)
    cv2.rectangle(image, (x, y), (x + fill_w, y + h), color, -1)

# Colored bounding boxes around features
def draw_tracking_rect(image, landmarks, color):
    np_landmarks = np.array(landmarks, dtype=np.int32)
    x, y, w, h = cv2.boundingRect(np_landmarks)
    pad = 5
    cv2.rectangle(image, (x - pad, y - pad), (x + w + pad, y + h + pad), color, 1)

# Setup MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

# Initialize Alarm
pygame.mixer.init()
alarm_sound = pygame.mixer.Sound("alarm.wav")

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
SLEEPY_BLINK_TIME = 0.5
BLINK_RATE_THRESHOLD = 20
RECOVERY_TIME = 15
MAR_THRESHOLD = 0.6
YAWN_TIME = 1.5

DISTRACT_TIME = 2.0
DISTRACT_RECOVERY_TIME = 0.5
PANEL_WIDTH = 550

# Runtime Variables
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

# Fatigue Score Tracker
fatigue_score = 0.0
fatigue_level = "SAFE"
last_fatigue_update = time.time()
drowsy_active = False
last_logged_state = None

# Video Evidence Settings
EVIDENCE_DIR = "evidence"
if not os.path.exists(EVIDENCE_DIR):
    os.makedirs(EVIDENCE_DIR)

recording = False
video_writer = None
record_start_time = None
RECORD_DURATION = 5

# CSV Logging Setup
LOG_FILE = "driver_log.csv"
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Date", "Time", "State", "FatigueScore"])

cap = cv2.VideoCapture(0)

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
                            video_path = f"{EVIDENCE_DIR}/drowsy_{timestamp}.mp4"
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

            # Log events to CSV
            if state != last_logged_state:
                now = time.localtime()
                date_str = time.strftime("%Y-%m-%d", now)
                time_str = time.strftime("%H:%M:%S", now)

                with open(LOG_FILE, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([
                        date_str,
                        time_str,
                        state,
                        int(fatigue_score)
                    ])

                last_logged_state = state

            # Draw UI
            draw_tracking_rect(frame, left_eye, state_color)
            draw_tracking_rect(frame, right_eye, state_color)
            draw_tracking_rect(frame, mouth, state_color)

            # Render Dashboard
            combined_display = np.zeros((h, w + PANEL_WIDTH, 3), dtype=np.uint8)
            combined_display[:] = COLOR_BG # Dark background
            combined_display[:h, :w] = frame

            # Offset for panel
            px = w + 20
            py = 40

            # 1. HEADER
            cv2.putText(combined_display, "DRIVER MONITOR", (px, py), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_TEXT, 2)
            
            # Live Time
            live_time = time.strftime("%H:%M:%S")
            cv2.putText(combined_display, live_time, (w + PANEL_WIDTH - 150, py), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_ACCENT, 2)

            cv2.line(combined_display, (px, py + 10), (w + PANEL_WIDTH - 20, py + 10), COLOR_ACCENT, 2)
            py += 50

            # 2. STATUS CARD
            status_color = state_color
            cv2.rectangle(combined_display, (px, py), (w + PANEL_WIDTH - 20, py + 60), status_color, -1)
            cv2.putText(combined_display, state, (px + 10, py + 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,0), 3)
            py += 90

            # 3. EYE TRACKING SECTION
            cv2.putText(combined_display, "Eye Tracking", (px, py), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT, 1)
            py += 20
            
            # EAR Bar
            ear_color = COLOR_OK if ear > EAR_THRESHOLD else COLOR_DANGER
            draw_bar(combined_display, px, py, 300, 15, ear, 0.4, ear_color)
            cv2.putText(combined_display, f"EAR: {ear:.2f}", (px + 310, py + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT, 1)
            py += 30

            # Blink Rate
            cv2.putText(combined_display, f"Blinks/min: {blink_rate}", (px, py), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_ACCENT, 2)
            py += 30
            # Yawns
            cv2.putText(combined_display, f"Yawns: {yawn_count}", (px, py), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_WARN, 2)
            py += 50

            # 4. HEAD POSTURE SECTION
            cv2.putText(combined_display, "Head Posture", (px, py), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT, 1)
            py += 25

            cv2.putText(combined_display, f"Direction: {direction}", (px, py), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            py += 40
            
            # Simple Text Stats for angles
            cv2.putText(combined_display, f"Nose Offset: {nose_offset:.3f}", (px, py), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
            py += 25
            cv2.putText(combined_display, f"Yaw: {int(yaw)}  Pitch: {int(pitch)}", (px, py), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
            
            # 5. FATIGUE SCORE SECTION
            py += 40
            cv2.putText(combined_display, "FATIGUE SCORE", (px, py), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT, 1)
            py += 20

            bar_color = COLOR_OK if fatigue_score < 30 else COLOR_WARN if fatigue_score < 60 else COLOR_DANGER
            draw_bar(combined_display, px, py, 300, 18, fatigue_score, 100, bar_color)

            cv2.putText(combined_display, f"{int(fatigue_score)}% - {fatigue_level}", (px + 310, py + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_TEXT, 1)


            cv2.imshow("Driver Monitoring System", combined_display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
