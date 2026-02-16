import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2, time
import numpy as np
import pyvirtualcam



model_path = "./hand_landmarker.task"

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

index_finger_trajectory = []  # List to store index finger positions

# Create a hand landmarker instance with the live stream mode:
def print_result(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global index_finger_trajectory

    if result.hand_landmarks:
        hand = result.hand_landmarks[0]

        index_finger = hand[8]  # Index finger tip is landmark 8

        frame_np = output_image.numpy_view().copy()  # or mp_image.numpy() depending on version
        h, w, _ = frame_np.shape

        x_px = int(index_finger.x * w)
        y_px = int(index_finger.y * h)

        index_finger_trajectory.append((x_px, y_px))

        for i in range(1, len(index_finger_trajectory)):
            cv2.line(frame_np, index_finger_trajectory[i-1], index_finger_trajectory[i], (0, 255, 0), 2)

        cv2.imshow('Hand Landmarker', frame_np)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return

    else:
        # If no hand is detected, clear the trajectory
        index_finger_trajectory = []
        frame_np = output_image.numpy_view().copy()  # or mp_image.numpy() depending on version
        cv2.imshow('Hand Landmarker', frame_np)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return





# Create landmarker options
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_hands=1,
    result_callback=print_result
)

# Open webcam
cap = cv2.VideoCapture(1)

# with pyvirtualcam.Camera(width=640, height=480, fps=30) as cam:
with HandLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break


        frame = cv2.flip(frame, 1)

        # Convert to MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

        # Send frame to hand landmarker (async)
        timestamp_ms = int(time.time() * 1000)
        landmarker.detect_async(mp_image, timestamp_ms)



cap.release()
cv2.destroyAllWindows()