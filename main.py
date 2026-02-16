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

        frame_np = output_image.numpy_view().copy()
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
        frame_np = output_image.numpy_view().copy()
        cv2.imshow('Hand Landmarker', frame_np)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return



# Create landmarker options
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO, # Live stream mode?
    num_hands=1,
    # result_callback=print_result
)

# Open webcam
cap = cv2.VideoCapture(0)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = 30

with pyvirtualcam.Camera(width=width, height=height, fps=fps) as cam:
    with HandLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break


            frame = cv2.flip(frame, 1)

            # Convert to MediaPipe Image
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

            timestamp_ms = int(time.time() * 1000)

            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            # Draw on frame
            if result.hand_landmarks:
                hand = result.hand_landmarks[0]
                index_finger = hand[8]

                h, w, _ = frame.shape
                x_px = int(index_finger.x * w)
                y_px = int(index_finger.y * h)

                index_finger_trajectory.append((x_px, y_px))

                for i in range(1, len(index_finger_trajectory)):
                    cv2.line(frame,
                             index_finger_trajectory[i - 1],
                             index_finger_trajectory[i],
                             (0, 255, 0), 2)
            else:
                index_finger_trajectory.clear()

            # Convert BGR → RGB for virtual cam
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            cam.send(frame_rgb)
            cam.sleep_until_next_frame()

