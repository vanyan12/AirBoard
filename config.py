import mediapipe as mp


# Configuration file for the hand gesture recognition application
model_path = "./hand_landmarker.task"


BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
# HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode


# Gesture recognition configuration
STABLE_FRAMES = 4
MIN_DIST = 3

# Draw and erase stability counters
drawing_stable_counter = 0
not_drawing_stable_counter = 0
erasing_stable_counter = 0
not_erasing_stable_counter = 0

drawing_active = False

# Pen parameters and button configuration
colors = [
    (255, 0, 0),   # Blue
    (0, 255, 0),   # Green
    (0, 0, 255),   # Red
]

PEN_SIZE = 7

BUTTON_RECT = (20, 200, 100, 250)
PEN_COLOR = colors[0]

ERASER_RADIUS = 25