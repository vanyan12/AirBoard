import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2, time
import numpy as np
import math

from state import DrawingState, Mode
from utils import *
from config import *
import pyvirtualcam

state = DrawingState()
state.pen_color = PEN_COLOR

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

# Set camera resolution (change 640, 480 to your desired width and height)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = 30

# with pyvirtualcam.Camera(width=width, height=height, fps=fps) as cam:
with HandLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # frame = cv2.flip(frame, 1)

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
            is_color_change_gesture = False #change_color(hand) # TODO: add color change gesture
            eraser_active = is_erasing(hand)

            # Color change logic
            if is_color_change_gesture:
                state.pen_color = colors[(colors.index(state.pen_color) + 1) % len(colors)]
                time.sleep(0.5)  # Add a small delay to prevent rapid color changes

            # Gesture stability logic for drawing
            if is_drawing_gesture:
                state.drawing_stable_counter += 1
                state.not_drawing_stable_counter = 0
            else:
                state.not_drawing_stable_counter += 1
                state.drawing_stable_counter = 0


            # Start drawing after stable frames
            if state.drawing_stable_counter >= STABLE_FRAMES and state.mode != Mode.DRAWING:
                state.mode = Mode.DRAWING

            # End drawing after stable non-drawing frames
            if state.not_drawing_stable_counter >= STABLE_FRAMES and state.mode == Mode.DRAWING:
                state.end_stroke()
                state.mode = Mode.IDLE


            if eraser_active:
                # Eraser mode overrides drawing
                if state.mode == Mode.DRAWING:
                    state.end_stroke()

                state.mode = Mode.ERASING

                eraser = eraser_center(hand, w, h)
                cv2.circle(frame, eraser, ERASER_RADIUS, (255, 255, 255), 3)
                state.drawing_segments = erase_segments(state.drawing_segments, eraser)

            # Change mode when eraser is deactivated
            elif state.mode == Mode.ERASING:
                state.mode = Mode.IDLE


            # ACTUAL DRAWING ===========================================================================
            if state.mode == Mode.DRAWING:
                if not state.current_segment:
                    state.add_point(smooth_point(x_px, y_px, None))
                else:
                    last_x, last_y = state.current_segment[-1]

                    # Calculates euclidean distance between last point and current point
                    distance = math.hypot(x_px - last_x, y_px - last_y)

                    if distance > MIN_DIST:
                        state.add_point(smooth_point(x_px, y_px, state.current_segment[-1]))


        else:
            # Hand lost: commit active stroke once and reset stability state
            state.on_hand_lost()

        continue_drawing(frame, state.drawing_segments, state.current_segment)

        # frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        cv2.imshow("Virtual Drawing", frame)
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
            # cam.send(frame_rgb)
            # cam.sleep_until_next_frame()

