# Drowsy Driver Detection System

A production-ready, real-time computer vision system that monitors driver fatigue using MediaPipe and OpenCV. This project detects drowsiness through multiple physiological indicators: Eye Aspect Ratio (EAR), Blink Rate, and Head Pose estimation.

## Features

*   **Real-time Monitoring**: Uses webcam feed to analyze driver state.
*   **Multi-Factor Detection**:
    *   **Eye Closure**: Monitors Eye Aspect Ratio (EAR) to detect prolonged eye closure.
    *   **Blink Rate**: Analyzes blinks per minute (fatigue < 10/min, strain > 30/min).
    *   **Head Pose**: Detects head nodding (pitch > 20 degrees).
*   **Alert System**: Visual (Screen Overlay) and Audio (Alarm Sound) warnings.
*   **Live Dashboard**: Displays real-time metrics, graphs, and status.
*   **Data Logging**: Records drowsiness events with timestamps to CSV.
*   **Robustness**: Handles multiple faces (warnings) and no-face scenarios.

## Project Structure

```
drowsy_driver_detection/
├── src/
│   ├── camera/           # Video capture & FPS limiting
│   ├── detection/        # Core logic (FaceMesh, EAR, Blink, Pose)
│   ├── alerts/           # Audio/Visual alarm system
│   ├── dashboard/        # UI rendering
│   └── utils/            # Constants & helpers
├── assets/
│   └── alarm.wav         # Alert sound
├── main.py               # Entry point
├── requirements.txt      # Dependencies
└── README.md             # Documentation
```

## Installation

1.  **Prerequisites**: Python 3.8+

2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Hardware**:
    *   Webcam (Built-in or USB)
    *   Speakers (for audio alerts)

## Usage

Run the main script:

```bash
python main.py
```

### Controls
*   **'q'**: Quit the application.

### Arguments
*   `--output`: Path to the CSV log file (default: `drowsiness_log.csv`).

Example:
```bash
python main.py --output my_trip_log.csv
```

## Technical Details

### Detection Logic
The system triggers a **DROWSY** alert if any of the following conditions are met:
1.  **EAR < 0.25** for more than **2.5 seconds** (Eyes closed).
2.  **Head Pitch > 20°** (downward) for more than **2.0 seconds** (Nodding).
3.  **Blink Rate** is outside the normal range (10-30 blinks/min).

### Technologies Used
*   **OpenCV**: Image processing and drawing.
*   **MediaPipe**: High-fidelity facial landmark detection (468 points).
*   **NumPy/SciPy**: Mathematical calculations (Euclidean distance, matrix operations).
*   **Pygame**: Asynchronous audio playback.

## Troubleshooting

*   **"Failed to open camera"**: Ensure no other application is using the webcam. Check the `CAMERA_INDEX` in `src/utils/constants.py` if you have multiple cameras.
*   **Audio not playing**: Check system volume and ensure `assets/alarm.wav` exists.
*   **False Positives**: 
    *   Ensure good lighting on the face.
    *   Adjust thresholds in `src/utils/constants.py` (e.g., `EAR_THRESHOLD`, `PITCH_THRESHOLD`) to fit your specific setup.
*   **Lag**: The system limits FPS to 30 to save resources. If it's too slow, check your CPU usage.

## License
MIT License
