from tkinter.constants import BUTT

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2, time
import numpy as np
import math
from utils import *
from config import *


drawing_segments = []  # List to store segments of the drawing | {"points": [(x1, y1), (x2, y2), ...], "color": (r, g, b)}
current_segment = []  # List to store the current segment being drawn | (x, y)

# Create landmarker options
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO, # Live stream mode?
    num_hands=1,
    # result_callback=print_result
)

# Open webcam
cap = cv2.VideoCapture(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


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
            draw_toolpad(frame, PEN_COLOR, PEN_SIZE)

            hand = result.hand_landmarks[0]
            index_finger_tip = hand[8]


            h, w, _ = frame.shape
            x_px = int(index_finger_tip.x * w)
            y_px = int(index_finger_tip.y * h)

            is_drawing_gesture = activate_drawing(hand)
            is_color_change_gesture = False #change_color(hand) # TODO: add color change gesture
            eraser_active = is_erasing(hand)

            # Color change logic
            if is_color_change_gesture:
                PEN_COLOR = colors[(colors.index(PEN_COLOR) + 1) % len(colors)]
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
                    drawing_segments.append({"points": current_segment, "color": PEN_COLOR})

                current_segment = []

            if eraser_active:
                # Eraser mode
                drawing_active = False
                eraser = eraser_center(hand, w, h)
                cv2.circle(frame, eraser, ERASER_RADIUS, (255, 255, 255), 3)
                drawing_segments= erase_segments(drawing_segments, eraser)
                current_segments = []



            # ACTUAL DRAWING ===========================================================================
            if drawing_active:
                if not current_segment:
                    current_segment.append(smooth_point(x_px, y_px, None))
                else:
                    last_x, last_y = current_segment[-1]
                    distance = math.hypot(x_px - last_x, y_px - last_y) # Calculates euclidean distance between last point and current point

                    if distance > MIN_DIST:
                        current_segment.append(smooth_point(x_px, y_px, current_segment[-1]))


            continue_drawing(frame, drawing_segments, current_segment)

        else:
            not_drawing_stable_counter += 1

            drawing_stable_counter = 0

            if not_drawing_stable_counter == STABLE_FRAMES:

                drawing_active = False

                if len(current_segment) > 1:
                    drawing_segments.append({"points": current_segment, "color": PEN_COLOR})

                current_segment = []

            continue_drawing(frame, drawing_segments, current_segment)

        cv2.imshow('Hand Landmarker', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

