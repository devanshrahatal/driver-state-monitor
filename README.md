# Driver State Monitoring System

A real-time, computer vision-based system designed to enhance road safety by monitoring driver attentiveness. This project utilizes MediaPipe Face Mesh and OpenCV to detect signs of drowsiness, fatigue, and distraction.

## Key Features

- **Drowsiness Detection**: Monitors Eye Aspect Ratio (EAR) to detect prolonged eye closure.
- **Fatigue Analysis**: Calculates a comprehensive "Fatigue Score" (0-100%) based on blink rate, yawning, and drowsiness history.
- **Distraction Detection**: Tracks head pose (Yaw/Pitch) and lateral movement to alert when the driver looks away.
- **Yawn Detection**: Measures Mouth Aspect Ratio (MAR) to identify yawning as an early sign of fatigue.
- **Evidence Recording**: automatically records 5-second video clips of drowsiness events to the `evidence/` folder.
- **Event Logging**: Logs all state changes and fatigue scores to `driver_log.csv` for post-drive analysis.
- **Visual Dashboard**: Displays real-time metrics, warnings, and detailed statistics on a modern side panel.
- **Audio Alerts**: Plays an alarm sound when critical drowsiness or distraction is detected.

## Requirements

- Python 3.8 – 3.11
- Webcam
- MediaPipe
- OpenCV
- NumPy
- SciPy
- Pygame
- absl-py

## Installation

1.  **Clone the Repository** (or download the source code):

    ```bash
    git clone https://github.com/devanshrahatal/driver-state-monitor.git
    cd driver-state-monitor
    ```

2.  **Install Dependencies**:
    Ensure you have Python installed (3.8+ recommended). Install the required libraries using pip:

    ```bash
    pip install -r requirements.txt
    ```

    _Dependencies include: `opencv-python`, `mediapipe`, `numpy`, `scipy`, `pygame`, `absl-py`_

3.  **Alarm Sound**:
    Ensure an `alarm.wav` file is present in the project root. (You can replace it with any .wav file).

## Usage

### Run the Project

```bash
python driver_state_monitor.py
```

### Controls

- Press **Q** to quit the application.

### Quick Start Guide (How to Test)

1.  **Sit in front of the webcam**: Ensure your face is clearly visible and well-lit.
2.  **Run the script**: Execute the command above.
3.  **Test the features**:
    - **Close eyes** for 2 seconds -> _Triggers Drowsy Alarm & Recording_.
    - **Blink fast** repeatedly -> _Triggers Sleepy Warning_.
    - **Yawn** with mouth open -> _Increases Yawn Counter_.
    - **Turn head** left/right -> _Triggers Distraction Alert_.

## Output Data

- **Evidence Videos**: Saved in the `/evidence` folder (Time-stamped .mp4 clips of drowsy events).
- **Event Logs**: Saved in `driver_log.csv` (Date, Time, State, FatigueScore).

## Logic & Thresholds

The system uses specific metrics to determine the driver's state:

- **EAR (Eye Aspect Ratio)**: Threshold `< 0.23`. Eyes closed for `> 2.0s` triggers "DROWSY".
- **MAR (Mouth Aspect Ratio)**: Threshold `> 0.6`. Identifying large mouth openings (Yawns).
- **Blink Rate**: High frequency (`> 20 blinks/min`) indicates "SLEEPY" behavior.
- **Head Pose**: Looking away (Left/Right/Down) for `> 2.0s` triggers "DISTRACTED".
- **Fatigue Score**: weighted sum of:
  - Drowsy duration (+40/s)
  - Sleepy state (+20/s)
  - High Blink Rate (+15/s)
  - Yawning (+10 per event)
  - Distraction (+10/s)
  - _Recovers gradually when in ACTIVE state._

## File Structure

- `driver_state_monitor.py`: Main application logic.
- `requirements.txt`: Python dependencies.
- `evidence/`: Directory where auto-recorded evidence clips are saved.
- `driver_log.csv`: Log file storing timestamped driver states.
- `alarm.wav`: Audio alert file.

## Disclaimer

This project is for **educational and research purposes only**. It is not a certified safety device.

## Author

**Devansh Rahatal**
**GitHub:** [DevanshRahatal](https://github.com/DevanshRahatal)
**LinkedIn:** [Devansh Rahatal](https://www.linkedin.com/in/devansh-rahatal/)
**Email:** devanshrahatal@gmail.com


