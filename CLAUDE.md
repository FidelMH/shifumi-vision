# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

IN THIS PROJECT, DON'T EVER USE EMOJIS
## Project Overview

This is a real-time hand gesture recognition game (Rock-Paper-Scissors / "Shifumi") built using:
- **Streamlit** with `streamlit-webrtc` for the web interface and webcam streaming
- **MediaPipe Hands** for hand landmark detection
- **OpenCV** for image processing
- **NumPy** for geometric calculations (angles, distances)

The application uses computer vision to detect hand gestures in real-time from webcam input and classify them as Rock (✊), Paper (✋), or Scissors (✌️).

## Development Commands

### Running the Application
```bash
streamlit run main.py
```

### Installing Dependencies
```bash
pip install -r requirements.txt
```

Note: The project uses a virtual environment located in `.venv/`

### Testing GPU Availability
The app checks for GPU availability at startup for potential MediaPipe optimization.

## Architecture & Key Components

### Core Class: `ImprovedHandDetector`

Located in `main.py`, this class extends `VideoProcessorBase` from `streamlit-webrtc` and handles the entire gesture detection pipeline:

1. **Initialization** (`__init__`):
   - Configures MediaPipe Hands with `max_num_hands=1` (currently set to 1, but recent commits show it was increased to 2)
   - Sets confidence thresholds: `min_detection_confidence=0.7`, `min_tracking_confidence=0.7`
   - Initializes a gesture history deque (size 10) for temporal stabilization

2. **Frame Processing** (`recv`):
   - Converts incoming video frames to BGR format
   - Processes frames through MediaPipe for hand landmark detection
   - Draws hand landmarks and connections on the frame
   - Calls gesture detection and displays results with confidence levels
   - Returns processed frames back to the stream

3. **Gesture Detection Logic** (`detect_gesture_improved`):
   - Converts normalized landmarks to pixel coordinates
   - Uses geometric analysis (angles, distances) to classify gestures
   - Returns gesture label and confidence score
   - **Rock**: All fingers closed (detected via `is_palm_closed` or `fingers_up == 0`)
   - **Scissors**: Exactly 2 fingers extended (index and middle), with distance check between them
   - **Paper**: 4-5 fingers extended
   - **Unknown**: Any other configuration returns "??? 🤔" with low confidence

4. **Helper Methods**:
   - `count_fingers_improved`: Counts extended fingers using angle-based detection
   - `is_finger_extended`: Checks if a finger is extended by comparing tip-to-MCP vs PIP-to-MCP distances
   - `is_thumb_extended`: Special logic for thumb detection (uses distance from thumb tip to index MCP)
   - `is_palm_closed`: Calculates average distance of all fingertips to wrist to detect closed fist

### Gesture Stabilization

The app uses a 10-frame rolling history (`deque(maxlen=10)`) to stabilize gesture detection. The displayed gesture is the most frequent gesture in this history, reducing flickering from frame-to-frame variations.

### Visual Feedback

- **Green text** (confidence > 80%): High confidence detection
- **Orange text** (confidence ≤ 80%): Lower confidence detection
- **Red text**: "Montrez votre main" when no hand is detected

## Important Implementation Details

### Coordinate System
- MediaPipe returns normalized coordinates (0-1 range)
- These are converted to pixel coordinates using frame width/height
- Y-axis: Lower values are higher on screen (tip[1] < pip[1] means finger is extended upward)

### Threshold Values
Current hardcoded thresholds that may need tuning:
- Index-middle finger distance for scissors: `< 100` pixels
- Thumb extension distance: `> 50` pixels
- Palm closed average distance: `< 120` pixels
- Finger extension ratio: `distance_tip_mcp > distance_pip_mcp * 1.2`

### Branch Information
- Main branch: `main`
- Current development branch: `poc-hand-recognition`

## Known Configuration

- The MediaPipe Hands configuration shows `max_num_hands=1` in the current code, but git history indicates it was previously set to 2
- The app includes GPU availability checking but doesn't currently utilize GPU acceleration explicitly
