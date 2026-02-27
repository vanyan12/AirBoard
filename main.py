from tkinter.constants import BUTT

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2, time
import numpy as np
import math
from utils import *



model_path = "./hand_landmarker.task"

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode


drawing_segments = []  # List to store segments of the drawing
current_segment = []  # List to store the current segment being drawn


STABLE_FRAMES = 5
MIN_DIST = 5

drawing_stable_counter = 0
not_drawing_stable_counter = 0
erasing_stable_counter = 0
not_erasing_stable_counter = 0

drawing_active = False

colors = [
    (255, 0, 0),   # Blue
    (0, 255, 0),   # Green
    (0, 0, 255),   # Red
]
BUTTON_RECT = (200, 20, 250, 40)
color = colors[0]



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

    for segment in drawing_segments:
        if len(segment['points']) > 1:
            pts = np.array(segment['points'], dtype=np.int32)
            cv2.polylines(frame,
                          [pts],          # must be list of arrays
                          isClosed=False, # IMPORTANT for drawing strokes
                          color=segment['color'],
                          thickness=5,
                          lineType=cv2.LINE_AA)

    if len(current_segment) > 1:
        pts = np.array(current_segment, dtype=np.int32)
        cv2.polylines(frame,
                      [pts],          # must be list of arrays
                      isClosed=False, # IMPORTANT for drawing strokes
                      color=color,
                      thickness=5,
                      lineType=cv2.LINE_AA)




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

            is_drawing_gesture = activate_drawing(hand)
            is_color_change_gesture = change_color(hand)

            if is_color_change_gesture:
                color = colors[(colors.index(color) + 1) % len(colors)]
                time.sleep(0.5)  # Add a small delay to prevent rapid color changes

            # GESTURE STABILITY LOGIC for drawing
            if is_drawing_gesture:
                drawing_stable_counter += 1
                not_drawing_stable_counter = 0
            else:
                not_drawing_stable_counter += 1
                drawing_stable_counter = 0


            # START DRAWING (stable 5 frames)
            if drawing_stable_counter == STABLE_FRAMES and not drawing_active:
                drawing_active = True

            # END DRAWING (stable 5 frames)
            if not_drawing_stable_counter == STABLE_FRAMES:
                drawing_active = False


                if len(current_segment) > 1:
                    drawing_segments.append({"points": current_segment, "color": color})

                current_segment = []



            # ACTUAL DRAWING
            if drawing_active:
                if not current_segment:
                    current_segment.append(smooth_point(x_px, y_px, None))
                else:
                    last_x, last_y = current_segment[-1]
                    distance = math.hypot(x_px - last_x, y_px - last_y)

                    if distance > MIN_DIST:
                        current_segment.append(smooth_point(x_px, y_px, current_segment[-1]))


            continue_drawing(frame, current_segment)

        else:
            not_drawing_stable_counter += 1

            drawing_stable_counter = 0

            if not_drawing_stable_counter == STABLE_FRAMES:

                drawing_active = False

                if len(current_segment) > 1:
                    drawing_segments.append({"points": current_segment, "color": color})

                current_segment = []

            continue_drawing(frame, current_segment)
        x1, y1, x2, y2 = BUTTON_RECT
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)

        cv2.imshow('Hand Landmarker', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

