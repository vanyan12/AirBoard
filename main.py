import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import time
import os
import numpy as np
import utils as u  # Make sure utils.py has activate_eraser()

# Model path
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "hand_landmarker.task")

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Drawing storage per hand
drawing_segments = {0: [], 1: []}       # completed strokes (list of (points, color))
current_segments = {0: [], 1: []}       # current stroke per hand
current_colors = {0: (0, 255, 0), 1: (0, 255, 0)}  # color of current segment per hand

# Eraser radius in pixels
ERASER_RADIUS = 30

def show_connections(frame, hand_landmarks):
    h, w, _ = frame.shape
    for landmark in hand_landmarks:
        x_px = int(landmark.x * w)
        y_px = int(landmark.y * h)
        cv2.circle(frame, (x_px, y_px), 4, (0, 0, 255), -1)

def eraser_point(hand, frame_width, frame_height):
    # Average of all finger tips (thumb + index + middle + ring + pinky)
    points = [hand[4], hand[8], hand[12], hand[16], hand[20]]
    x = int(sum([p.x for p in points]) / len(points) * frame_width)
    y = int(sum([p.y for p in points]) / len(points) * frame_height)
    return (x, y)

def erase_segments(segments, eraser_pos, radius=ERASER_RADIUS):
    # Remove segments that have a point within radius of eraser
    new_segments = []
    for segment, color in segments:
        erase = False
        for px, py in segment:
            if (px - eraser_pos[0])**2 + (py - eraser_pos[1])**2 <= radius**2:
                erase = True
                break
        if not erase:
            new_segments.append((segment, color))
    return new_segments

def recognize_shape(segment):
    """Return either modified shape (circle/rectangle) or keep freehand"""
    if len(segment) < 5:
        return segment, None  # too few points, keep freehand

    pts = np.array(segment, dtype=np.int32)

    # Circle check
    (x, y), radius = cv2.minEnclosingCircle(pts)
    center = (int(x), int(y))
    radius = int(radius)
    perimeter = cv2.arcLength(pts, False)
    area = cv2.contourArea(pts)
    circularity = 4 * np.pi * area / (perimeter**2) if perimeter > 0 else 0

    if 0.6 < circularity <= 1.2:
        return [], ("circle", center, radius)

    # Rectangle check → axis-aligned bounding rectangle
    x, y, w, h = cv2.boundingRect(pts)
    box = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.int32)
    rect_area = w * h
    rectangularity = area / rect_area if rect_area > 0 else 0
    if 0.6 < rectangularity <= 1.2:
        return [], ("rectangle", box)

    # Otherwise keep freehand
    return segment, None

def continue_drawing(frame):
    for hand_id in [0, 1]:
        # Draw completed segments
        for segment, color in drawing_segments[hand_id]:
            seg, shape = recognize_shape(segment)
            if shape:
                if shape[0] == "circle":
                    cv2.circle(frame, shape[1], shape[2], color, 2)
                elif shape[0] == "rectangle":
                    cv2.polylines(frame, [shape[1]], True, color, 2)
            else:
                for i in range(1, len(seg)):
                    cv2.line(frame, seg[i-1], seg[i], color, 2)

        # Draw current segment
        for i in range(1, len(current_segments[hand_id])):
            cv2.line(frame, current_segments[hand_id][i-1], current_segments[hand_id][i], current_colors[hand_id], 2)

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2,
)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

with HandLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        timestamp_ms = int(time.time() * 1000)

        result = landmarker.detect_for_video(mp_image, timestamp_ms)
        h, w, _ = frame.shape

        if result.hand_landmarks:
            for idx, hand in enumerate(result.hand_landmarks):
                show_connections(frame, hand)

                green_active = u.activate_drawing(hand)
                red_active = u.activate_red_drawing(hand)
                eraser_active = u.activate_eraser(hand)

                if eraser_active:
                    # Eraser mode
                    point = eraser_point(hand, w, h)
                    cv2.circle(frame, point, ERASER_RADIUS, (255, 255, 255), 2)
                    drawing_segments[idx] = erase_segments(drawing_segments[idx], point)
                    current_segments[idx] = []
                else:
                    # Green = index
                    if green_active and not red_active:
                        finger_tip = hand[8]
                        x_px = int(finger_tip.x * w)
                        y_px = int(finger_tip.y * h)
                        current_segments[idx].append((x_px, y_px))
                        current_colors[idx] = (0, 255, 0)
                    # Red = pinky
                    elif red_active and not green_active:
                        finger_tip = hand[20]
                        x_px = int(finger_tip.x * w)
                        y_px = int(finger_tip.y * h)
                        current_segments[idx].append((x_px, y_px))
                        current_colors[idx] = (0, 0, 255)
                    else:
                        # Stop current stroke
                        if current_segments[idx]:
                            drawing_segments[idx].append((current_segments[idx], current_colors[idx]))
                            current_segments[idx] = []

        else:
            # No hands detected → save ongoing strokes
            for idx in [0, 1]:
                if current_segments[idx]:
                    drawing_segments[idx].append((current_segments[idx], current_colors[idx]))
                    current_segments[idx] = []

        continue_drawing(frame)
        cv2.imshow("AirBoard Gesture Drawing", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
