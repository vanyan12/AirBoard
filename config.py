import mediapipe as mp


# Configuration for hand gesture recognition.
MODEL_PATH = "./hand_landmarker.task"

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
# HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode


# Gesture recognition thresholds.
STABLE_FRAMES = 3
MIN_DIST = 2

# Pen parameters and palette (B, G, R).
COLORS = [
    (255, 0, 0),  # Blue
    (0, 255, 0),  # Green
    (0, 0, 255),  # Red
]

DEFAULT_PEN_COLOR = COLORS[0]
DEFAULT_PEN_SIZE = 5
DEFAULT_ERASER_RADIUS = 25

BUTTON_RECT = (20, 200, 100, 250)
