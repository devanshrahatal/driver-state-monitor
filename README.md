# DriveGuard — Driver State Monitoring System

A real-time, computer vision-based safety system built to continuously monitor driver attentiveness. DriveGuard utilizes **MediaPipe Face Mesh** and **OpenCV** to dynamically detect signs of drowsiness, fatigue, and distraction, actively alerting the driver and initiating emergency protocols if they become unresponsive.

## Key Features

### Core Detection Metrics
- **Micro-Sleep & Drowsiness**: Continuously calculates the Eye Aspect Ratio (EAR) to detect prolonged eye closure.
- **Cognitive Fatigue Analysis**: Generates a real-time, weighted **Fatigue Score (0-100%)** integrating blink rate frequency, yawning occurrences, and cumulative drowsiness history.
- **Distraction Tracking**: Maps 3D head pose (Yaw/Pitch/Roll) and calculates nose-offset vectoring to trigger alerts when the driver looks away from the road.
- **Yawn Detection**: Monitors the Mouth Aspect Ratio (MAR) to catch yawning as an early fatigue indicator.

### Active Alerts & Emergency Protocols
- **Intelligent Audio Alarms**: High-visibility UI overlays accompanied by escalating localized alarms.
- **Automated Emergency Emails**: If a driver enters an unrecoverable unresponsive state (eyes continuously closed for >15 seconds), DriveGuard automatically dispatches an SOS email to a configured emergency contact via SMTP (repeating every 30 seconds).
- **Audio-Ducked Voice Announcements**: Features a non-blocking Text-To-Speech (TTS) engine that ducks (lowers) the active siren volume temporarily to clearly announce: *"Emergency email sent. Please pull over safely."*
- **Evidence Recording**: Automatically records and isolates 5-second video `.mp4` chunks of drowsiness events to the `evidence/` folder.

### Drive Analysis & PDF Reporting System
- **Detailed Session Logging**: Every drive natively exports a continuous 10-metric CSV (EAR, MAR, Blink Rate, Pose, Score, etc.) into the `records/` directory.
- **Post-Drive Analyzer**: Includes `session_analysis.py`, a dedicated forensic reporting tool.
- **Automated PDF Reports**: Transforms raw CSV session data into professional, print-ready PDF reports complete with a calculated **Overall Safety Grade (A-F)** and customized, plain-English behavioral recommendations.
- **Data Visualizations**: Automatically plots `matplotlib` charts (State Timelines, Fatigue Area Charts, Head Direction Pie Charts) saving them neatly inside `reports/analysis_session_.../charts/`.

### Premium UI Dashboard
- Deep Navy theme equipped with visually distinct components, floating panels, and typography.
- Statuses are distinctly color-coded (Active, Sleepy, Drowsy, Distracted).
- Features non-blocking asynchronous module startup (preventing camera stutter during ML initialization) driven by smooth loading spinners.

---

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/devanshrahatal/driver-state-monitor.git
   cd driver-state-monitor
   ```

2. **Install Dependencies**:
   Ensure you have Python 3.8 - 3.11 installed. (Note: Ensure protobuf compatibility if resolving MediaPipe).
   ```bash
   pip install -r requirements.txt
   ```
   _Core Dependencies: `opencv-python`, `mediapipe`, `numpy`, `pandas`, `matplotlib`, `fpdf2`, `pygame`, `pyttsx3`_

3. **Assets**:
   Ensure an `alarm.wav` file is present in the project root directory.

---

## Usage Guide

### 1. Start the Live Monitoring System
To boot up the camera and begin monitoring:
```bash
python driver_state_monitor.py
```
* **Controls**: Press **`Q`** to safely quit the application and finalize the log.

### 2. Generate a Post-Drive Analysis Report
After completing a drive session, navigate to the `records/` folder to locate your CSV log, then run:
```bash
python session_analysis.py
```
When prompted, paste the path to your session CSV (e.g., `records/session_2026-04-11_14-00-00.csv`). The script will generate your UI charts and output your final PDF into the `reports/` folder!

---

## Logic & Variable Thresholds

The system relies on dynamically fine-tuned metrics:
- **EAR (Eye Aspect Ratio)**: Threshold `< 0.23`. Eyes closed for `> 2.0s` triggers "DROWSY".
- **MAR (Mouth Aspect Ratio)**: Threshold `> 0.6` identifies yawning.
- **Blink Rate**: Elevated frequencies (`> 20 blinks/min`) flag the driver as "SLEEPY".
- **Head Pose**: Looking off-center for `> 2.0s` flags the driver as "DISTRACTED".
- **Fatigue Weighting**:
  - Drowsy states: +40/sec
  - Sleepy states: +20/sec
  - Distracted states: +10/sec
  - _Fatigue naturally dissipates when maintaining a secure "ACTIVE" state._

---

## File Structure

- `driver_state_monitor.py` — The core ML, UI, and live camera feed execution script.
- `session_analysis.py` — The forensics script for generating PDF reports & charts from CSVs.
- `records/` — Home for all automatically generated session CSV files.
- `reports/` — Output directory for your PDF safety reports and charts.
- `evidence/` — Directory containing `.mp4` video captures of critical drowsiness hits.

---

## Disclaimer
This project is for **educational and research purposes only** and should not replace certified automotive safety devices. 

---

## Author

**Devansh Rahatal**
* **GitHub:** [@DevanshRahatal](https://github.com/DevanshRahatal)
* **LinkedIn:** [Devansh Rahatal](https://www.linkedin.com/in/devansh-rahatal/)
* **Email:** devanshrahatal@gmail.com
