# Gesture-to-Command Mapping System (Simulation)

> Real-time hand gesture recognition that maps finger counts to drone commands — no hardware required.

---

## Overview

This project uses a standard webcam, OpenCV, and MediaPipe Hands to detect the number of raised fingers in real time and translate them into predefined drone control commands. A temporal stability filter eliminates flickering so commands only commit after the same gesture is held for 1 second.

---

## Demo Commands

| Fingers | Command        |
|---------|----------------|
| 0       | Idle           |
| 1       | Takeoff        |
| 2       | Hover          |
| 3       | Move Forward   |
| 4       | Move Backward  |
| 5       | Land           |

---

## Project Structure

```
Gesture_Command_System/
│
├── main.py                 ← Entry point; MainApplication class
├── hand_detector.py        ← MediaPipe wrapper + bounding box drawing
├── gesture_recognizer.py   ← Finger counting + temporal stability filter
├── command_mapper.py       ← Finger count → command string mapping
├── logger.py               ← Append-only event log writer
│
├── gesture_log.txt         ← Auto-created; one entry per command change
├── requirements.txt        ← Python dependencies
└── README.md               ← This file
```

---

## Setup

### 1 — Prerequisites

- Python 3.9 or 3.10 recommended (MediaPipe 0.10 supports up to 3.11)
- A working webcam

### 2 — Create a virtual environment (recommended)

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3 — Install dependencies

```bash
pip install -r requirements.txt
```

> **Note for Apple Silicon (M1/M2):** use `pip install mediapipe-silicon` instead of `mediapipe`.

### 4 — Run

```bash
python main.py
```

To specify a different camera index (e.g. external webcam on index 1):

```bash
python main.py 1
```

---

## Keyboard Controls

| Key | Action                              |
|-----|-------------------------------------|
| `Q` | Quit the application                |
| `R` | Reset command state to Idle         |
| `L` | Print last logged command + overlay |

---

## UI Layout

```
┌─────────────────────────────────────────────┬──────────────────────┐
│  [0] [1] [2] [3] [4] [5]   Command Received │  SYSTEM MONITOR      │
│─────────────────── BANNER ──────────────────│  Finger Count : 2    │
│                                             │  Command      : Hover│
│   (live webcam feed)                        │  Confidence   : 98%  │
│                                             │  FPS          : 30   │
│   ┌──────────────────────────┐              │  Status   : Tracking │
│   │  Hand with landmarks     │              │                      │
│   │  + bounding box          │              │  [Confidence bar]    │
│   └──────────────────────────┘              │                      │
│                                             │  COMMAND REFERENCE   │
│                                             │  ▶ 2  Hover          │
│                                             │    3  Move Forward   │
│                                             │  Q Quit  R Reset     │
└─────────────────────────────────────────────┴──────────────────────┘
```

---

## Log File

`gesture_log.txt` is created automatically in the project directory.  
Each command change is appended:

```
2026-06-07 12:45:01 | 2 Fingers | Hover
2026-06-07 12:45:08 | 1 Finger  | Takeoff
2026-06-07 12:45:15 | 5 Fingers | Land
```

---

## Architecture

| Class               | File                    | Responsibility                                      |
|---------------------|-------------------------|-----------------------------------------------------|
| `HandDetector`      | `hand_detector.py`      | Webcam frame → MediaPipe landmarks + bounding box   |
| `GestureRecognizer` | `gesture_recognizer.py` | Landmarks → finger count with stability filter      |
| `CommandMapper`     | `command_mapper.py`     | Finger count → command string + colour lookup       |
| `Logger`            | `logger.py`             | Append-only log writer with dedup                   |
| `MainApplication`   | `main.py`               | Orchestration, UI dashboard, keyboard controls      |

---

## Performance Notes

- Target ≥ 20 FPS at 1280×720; typically achieves 28–35 FPS on mid-range hardware.
- `max_num_hands=1` keeps MediaPipe processing lean.
- BGR→RGB conversion uses a writeable=False flag to avoid a memory copy.
- The stability window (`smoothing_window=10`) and hold time (`stability_seconds=1.0`) are constructor arguments and can be tuned.

---

## Troubleshooting

| Symptom                         | Fix                                                        |
|---------------------------------|------------------------------------------------------------|
| `Cannot open camera at index 0` | Try `python main.py 1` or check webcam is not in use       |
| Very low FPS                    | Lower resolution: edit `CAP_PROP_FRAME_WIDTH/HEIGHT`       |
| Finger count unstable           | Increase `stability_seconds` in `GestureRecognizer`        |
| MediaPipe import error          | Ensure Python ≤ 3.11 and re-run `pip install mediapipe`    |

---

## License

MIT — free to use, modify, and distribute.
