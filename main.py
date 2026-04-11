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
    COLORS,
)
from state import DrawingState, Mode
from utils import (
    activate_drawing,
    change_pen_color,
    change_color,
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
    color_change = change_color(hand)

    update_stability_counters(state, is_drawing_gesture)

    if eraser_active:
        apply_eraser(state, hand, frame, w, h)
    else:
        if state.mode == Mode.ERASING:
            state.mode = Mode.IDLE

        update_mode_from_gesture(state)
        apply_drawing_point(state, x_px, y_px)

    if color_change:
        state.pen_color = COLORS[(COLORS.index(state.pen_color) + 1) % len(COLORS)]
        time.sleep(0.5)  # debounce color change


def draw_toolbar(frame, state: DrawingState) -> None:
    h, w, _ = frame.shape
    toolbar_height = max(60, int(h * 0.1))

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, toolbar_height), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    cv2.putText(frame, "Color", (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (245, 245, 245), 2, cv2.LINE_AA)
    cv2.rectangle(frame, (90, 10), (150, 42), state.pen_color, -1)
    cv2.rectangle(frame, (90, 10), (150, 42), (245, 245, 245), 2)

    cv2.putText(frame, "Size", (175, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (245, 245, 245), 2, cv2.LINE_AA)
    cv2.putText(frame, str(state.pen_size), (240, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (245, 245, 245), 2, cv2.LINE_AA)
    cv2.circle(frame, (300, 26), max(2, state.pen_size), state.pen_color, -1)



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
            draw_toolbar(frame, state)

            cv2.imshow("Virtual Drawing", frame)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
