import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2, time
import numpy as np
import math
import utils as u



model_path = "./hand_landmarker.task"

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode


drawing_segments = []  # List to store segments of the drawing
current_segment = []  # List to store the current segment being drawn
not_drawing_frames = 0  # Counter for frames where drawing is not active

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

def show_connections(frame, hand_landmarks):

    h, w, _ = frame.shape
    for landmark in hand_landmarks:
        x_px = int(landmark.x * w)
        y_px = int(landmark.y * h)
        cv2.circle(frame, (x_px, y_px), 7, (0, 0, 255), -1)


    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (5, 9), (9, 10), (10, 11), (11, 12),
        (9, 13), (13, 14), (14, 15), (15, 16),
        (0,17), (13, 17), (17, 18), (18, 19), (19, 20)
    ]

    for connection in connections:
        start = hand_landmarks[connection[0]]
        end = hand_landmarks[connection[1]]

        x1, y1 = int(start.x * w), int(start.y * h)
        x2, y2 = int(end.x * w), int(end.y * h)

        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

def continue_drawing(frame, current_segment):
    # Draw the saved segments
    for segment in drawing_segments:
        for i in range(1, len(segment)):
            cv2.line(frame,
                     segment[i - 1],  # first point
                     segment[i],  # second point
                     (0, 255, 0), 2)

    for i in range(1, len(current_segment)):
        cv2.line(frame,
                 current_segment[i - 1],
                 current_segment[i],
                 (0, 255, 0), 2)




# Create landmarker options
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO, # Live stream mode?
    num_hands=1,
    # result_callback=print_result
)

# Open webcam
cap = cv2.VideoCapture(0) # change 0 to 1 if you have multiple cameras and want to use the second one

# Set camera resolution (change 640, 480 to your desired width and height)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)


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
            show_connections(frame, result.hand_landmarks[0])

            hand = result.hand_landmarks[0]
            index_finger_tip = hand[8]


            h, w, _ = frame.shape
            x_px = int(index_finger_tip.x * w)
            y_px = int(index_finger_tip.y * h)


            if u.activate_drawing(hand):
                not_drawing_frames = 0
                current_segment.append((x_px, y_px))

                continue_drawing(frame, current_segment)
            else:
                not_drawing_frames += 1
                # When the current segment have points and drawing mode is off, save the segment
                if current_segment:
                    drawing_segments.append(current_segment)
                    current_segment = []


        else:
            if len(current_segment) > 0:
                drawing_segments.append(current_segment)
                current_segment = []


        cv2.imshow('Hand Landmarker', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()





