# DriveGuard — Driver State Monitoring & Analyzing System

A real-time, computer vision-based safety system built to continuously monitor driver attentiveness. DriveGuard utilizes **MediaPipe Face Mesh** and **OpenCV** to dynamically detect signs of drowsiness, fatigue, and distraction, actively alerting the driver and initiating emergency protocols if they become unresponsive.

## Key Features

### Core Detection Metrics

- **Micro-Sleep & Drowsiness**: Continuously calculates the Eye Aspect Ratio (EAR) to detect prolonged eye closure.
- **Cognitive Fatigue Analysis**: Generates a real-time, weighted **Fatigue Score (0-100%)** integrating blink rate frequency, yawning occurrences, and cumulative drowsiness history.
- **Distraction Tracking**: Maps 3D head pose (Yaw/Pitch/Roll) and calculates nose-offset vectoring to trigger alerts when the driver looks away from the road.
- **Yawn Detection**: Monitors the Mouth Aspect Ratio (MAR) to catch yawning as an early fatigue indicator.

### Active Alerts & Emergency Protocols

- **Intelligent Audio Alarms**: High-visibility UI overlays accompanied by escalating localized alarms.
- **Session Chimes**: Professional audio feedback confirming session start and end.
- **Automated Break Reminders**: Voice-based notifications ("Please take a break and rest") triggered by a configurable session timer to prevent fatigue on long drives.
- **Automated Emergency Emails**: If a driver enters an unrecoverable unresponsive state (eyes continuously closed for >15 seconds), DriveGuard automatically dispatches an SOS email to a configured emergency contact via SMTP (repeating every 30 seconds).
- **Audio-Ducked Voice Announcements**: Features a non-blocking Text-To-Speech (TTS) engine that ducks (lowers) the active siren volume temporarily to clearly announce: _"Emergency email sent. Please pull over safely."_
- **Evidence Recording**: Automatically records and isolates 5-second video `.mp4` chunks of drowsiness events to the `evidence/` folder.

### Drive Analysis & PDF Reporting System

- **Detailed Session Logging**: Every drive natively exports a continuous 10-metric CSV (EAR, MAR, Blink Rate, Pose, Score, etc.) into the `records/` directory.
- **Post-Drive Analyzer**: Includes `session_analysis.py`, a dedicated forensic reporting tool.
- **AI-Powered Recommendations**: Integrates with **Google Gemini AI** to generate personalized, data-driven safety recommendations based on session metrics. Falls back to a comprehensive rule-based engine when AI is unavailable.
- **Automated PDF Reports**: Transforms raw CSV session data into professional, print-ready PDF reports complete with a calculated **Overall Safety Grade (A-F)** and customized, plain-English behavioral recommendations.
- **Data Visualizations**: Automatically plots `matplotlib` charts (State Timelines, Fatigue Area Charts, Head Direction Pie Charts) saving them neatly inside `reports/analysis_session_.../charts/`.

### Premium UI Dashboard

- Deep Navy theme equipped with visually distinct components, floating panels, and typography.
- Statuses are distinctly color-coded (Active, Sleepy, Drowsy, Distracted).
- **Real-Time Session Timer**: Features a dedicated tracking card with an integrated "Next Break" progress bar and remaining time countdown.
- Features non-blocking asynchronous module startup (preventing camera stutter during ML initialization) driven by smooth loading spinners.
- **Post-Session UI**: A sleek, modern Tkinter-based interface automatically prompts the user to generate an analytical report upon closing the monitor, supporting drag-and-drop CSV selection.

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

   _Core Dependencies: `opencv-python`, `mediapipe`, `numpy`, `pandas`, `matplotlib`, `fpdf2`, `pygame`, `pyttsx3`, `python-dotenv`_

3. **Configure Environment Secrets**:
   Create a `.env` file in the project root with your credentials:

   ```env
   DRIVEGUARD_SENDER_EMAIL=your_email@gmail.com
   DRIVEGUARD_SENDER_PASSWORD=your_app_password
   DRIVEGUARD_GEMINI_API_KEY=your_gemini_api_key
   ```

   > **Note**: The `.env` file is gitignored and never committed. Use a [Gmail App Password](https://support.google.com/accounts/answer/185833) for `DRIVEGUARD_SENDER_PASSWORD`. The Gemini API key is optional — if omitted, recommendations will use the built-in rule-based engine.

4. **Assets**:
   Ensure an `alarm.wav` file is present in the project root directory.

---

## Usage Guide

### 1. Start the Live Monitoring System

To boot up the camera and begin monitoring:

```bash
python driver_state_monitor.py
```

- **Controls**: Press **`Q`** to safely quit the application and finalize the log.

### 2. Generate a Post-Drive Analysis Report

After completing a drive session, navigate to the `records/` folder to locate your CSV log, then run:

```bash
python session_analysis.py
```

When prompted, paste the path to your session CSV (e.g., `records/session_2026-04-11_14-00-00.csv`). The script will generate your UI charts and output your final PDF into the `reports/` folder!

### 3. Run Unit Tests

```bash
pytest test_driveguard.py -v
```

---

## Configuration & Thresholds

DriveGuard uses a two-layer configuration system:

- **`config.json`** — All non-sensitive settings (thresholds, timers, UI options). Easily adjustable without modifying source code.
- **`.env`** — All sensitive credentials (SMTP email/password, API keys). Never committed to version control.

### Thresholds (`config.json`)

- **Eye Tracking (EAR)**: Threshold `< 0.23`. Eyes closed for `> 2.0s` triggers "DROWSY".
- **Mouth Tracking (MAR)**: Threshold `> 0.6` identifies yawning.
- **Blink Rate**: Elevated frequencies (`> 20 blinks/min`) flag the driver as "SLEEPY".
- **Head Pose**: Looking off-center for `> 2.0s` flags the driver as "DISTRACTED".
- **Session Config**: Configure break reminder intervals (e.g., 90 minutes) and evidence recording duration.
- **Emergency Config**: Emergency detection time and email repeat interval.

### Fatigue Weighting

- Drowsy states: +40/sec
- Sleepy states: +20/sec
- Distracted states: +10/sec
- _Fatigue naturally dissipates when maintaining a secure "ACTIVE" state._

---

## File Structure

- `driver_state_monitor.py` — The core ML, UI, and live camera feed execution script.
- `session_analysis.py` — The forensics script for generating PDF reports & charts from CSVs.
- `test_driveguard.py` — Unit tests for core analysis utilities.
- `config.json` — Non-sensitive configuration (thresholds, timers, session settings).
- `.env` — Sensitive credentials (SMTP, API keys). Gitignored.
- `.gitignore` — Excludes secrets, caches, and generated outputs from version control.
- `requirements.txt` — Python dependency list.
- `records/` — Home for all automatically generated session CSV files.
- `reports/` — Output directory for your PDF safety reports and charts.
- `evidence/` — Directory containing `.mp4` video captures of critical drowsiness hits.

---

## Disclaimer

This project is for **educational and research purposes only** and should not replace certified automotive safety devices.

---

## Author

**Devansh Rahatal**

- **GitHub:** [@DevanshRahatal](https://github.com/DevanshRahatal)
- **LinkedIn:** [Devansh Rahatal](https://www.linkedin.com/in/devansh-rahatal/)
- **Email:** devanshrahatal@gmail.com
