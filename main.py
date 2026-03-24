import math
import time

import cv2
import mediapipe as mp

from config import (
    BaseOptions,
    DEFAULT_ERASER_RADIUS,
    DEFAULT_PEN_COLOR,
    DEFAULT_PEN_SIZE,
    HandLandmarker,
    HandLandmarkerOptions,
    MIN_DIST,
    MODEL_PATH,
    STABLE_FRAMES,
    VisionRunningMode,
)
from state import DrawingState, Mode
from utils import (
    activate_drawing,
    draw_segments,
    erase_segments,
    eraser_center,
    is_erasing,
    show_connections,
    smooth_point,
)


def update_stability_counters(state: DrawingState, is_drawing_gesture: bool) -> None:
    if is_drawing_gesture:
        state.drawing_stable_counter += 1
        state.not_drawing_stable_counter = 0
    else:
        state.not_drawing_stable_counter += 1
        state.drawing_stable_counter = 0


def update_mode_from_gesture(state: DrawingState) -> None:
    if state.drawing_stable_counter >= STABLE_FRAMES and state.mode != Mode.DRAWING:
        state.mode = Mode.DRAWING
    if state.not_drawing_stable_counter >= STABLE_FRAMES and state.mode == Mode.DRAWING:
        state.end_stroke()
        state.mode = Mode.IDLE


def apply_drawing_point(state: DrawingState, x_px: int, y_px: int) -> None:
    if state.mode != Mode.DRAWING:
        return

    if not state.current_segment:
        state.add_point(smooth_point(x_px, y_px, None))
        return

    last_x, last_y = state.current_segment[-1]
    distance = math.hypot(x_px - last_x, y_px - last_y)
    if distance > MIN_DIST:
        state.add_point(smooth_point(x_px, y_px, state.current_segment[-1]))


def apply_eraser(state: DrawingState, hand, frame, frame_width: int, frame_height: int) -> None:
    if state.mode == Mode.DRAWING:
        state.end_stroke()

    state.mode = Mode.ERASING
    eraser = eraser_center(hand, frame_width, frame_height)
    cv2.circle(frame, eraser, state.eraser_radius, (255, 255, 255), 3)
    state.drawing_segments = erase_segments(state.drawing_segments, eraser, state.eraser_radius)


def process_hand(state: DrawingState, hand, frame) -> None:
    show_connections(frame, hand)

    h, w, _ = frame.shape
    index_finger_tip = hand[8]
    x_px = int(index_finger_tip.x * w)
    y_px = int(index_finger_tip.y * h)

    is_drawing_gesture = activate_drawing(hand)
    eraser_active = is_erasing(hand)

    update_stability_counters(state, is_drawing_gesture)

    if eraser_active:
        apply_eraser(state, hand, frame, w, h)
    else:
        if state.mode == Mode.ERASING:
            state.mode = Mode.IDLE

        update_mode_from_gesture(state)
        apply_drawing_point(state, x_px, y_px)


def main() -> None:
    state = DrawingState(
        pen_color=DEFAULT_PEN_COLOR,
        pen_size=DEFAULT_PEN_SIZE,
        eraser_radius=DEFAULT_ERASER_RADIUS,
    )

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=1,
    )

    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)  # request 30 FPS from camera

    with HandLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

            timestamp_ms = int(time.time() * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.hand_landmarks:
                process_hand(state, result.hand_landmarks[0], frame)
            else:
                state.on_hand_lost()

            draw_segments(
                frame,
                state.drawing_segments,
                state.current_segment,
                state.pen_color,
                state.pen_size,
            )

            cv2.imshow("Virtual Drawing", frame)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
