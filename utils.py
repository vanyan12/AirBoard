import numpy as np
import cv2
from config import *

# mcp - 0
# pip - 1
# dip - 2
# tip - 3
fingers = {
    'thumb': [1, 2, 3, 4],
    'index': [5, 6, 7, 8],
    'middle': [9, 10, 11, 12],
    'ring': [13, 14, 15, 16],
    'pinky': [17, 18, 19, 20],
    'wrist': [0]
}

def angle(j1, j2, j3):
    # j1 = np.array([j1.x, j1.y, j1.z])
    # j2 = np.array([j2.x, j2.y, j2.z])
    # j3 = np.array([j3.x, j3.y, j3.z])

    v1 = j1 - j2
    v2 = j3 - j2

    cos_ang = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    angle_rad = np.arccos(cos_ang)
    angle_deg = np.degrees(angle_rad)

    return angle_deg

def distance(p1, p2):
    return np.linalg.norm(p1 - p2)

def is_finger_straight(hand, finger_name, thr=170) -> bool:
    mcp = np.array([hand[fingers[finger_name][0]].x, hand[fingers[finger_name][0]].y, hand[fingers[finger_name][0]].z])
    pip = np.array([hand[fingers[finger_name][1]].x, hand[fingers[finger_name][1]].y, hand[fingers[finger_name][1]].z])
    dip = np.array([hand[fingers[finger_name][2]].x, hand[fingers[finger_name][2]].y, hand[fingers[finger_name][2]].z])
    tip = np.array([hand[fingers[finger_name][3]].x, hand[fingers[finger_name][3]].y, hand[fingers[finger_name][3]].z])

    angle_1 = angle(mcp, pip, dip)
    angle_2 = angle(pip, dip, tip)

    return angle_1 > thr and angle_2 > thr

def is_finger_bent(hand, finger_name, thr=165) -> bool:

    mcp = np.array([hand[fingers[finger_name][0]].x, hand[fingers[finger_name][0]].y, hand[fingers[finger_name][0]].z])
    pip = np.array([hand[fingers[finger_name][1]].x, hand[fingers[finger_name][1]].y, hand[fingers[finger_name][1]].z])
    dip = np.array([hand[fingers[finger_name][2]].x, hand[fingers[finger_name][2]].y, hand[fingers[finger_name][2]].z])
    tip = np.array([hand[fingers[finger_name][3]].x, hand[fingers[finger_name][3]].y, hand[fingers[finger_name][3]].z])

    angle_1 = angle(mcp, pip, dip)
    angle_2 = angle(pip, dip, tip)

    return angle_1 < thr or angle_2 < thr

# Exponential moving average (EMA) smoothing
def smooth_point(new_x, new_y, prev_point, alpha=0.4) -> tuple[float, float]:
    if prev_point is None:
        return float(new_x), float(new_y)

    prev_x, prev_y = prev_point
    smooth_x = alpha * new_x + (1 - alpha) * prev_x
    smooth_y = alpha * new_y + (1 - alpha) * prev_y
    return smooth_x, smooth_y

def is_index_highest(hand) -> bool:
    # y = 0 at top, y increases downward
    index_y = hand[8].y
    middle_y = hand[12].y
    ring_y = hand[16].y
    pinky_y = hand[20].y

    if index_y < middle_y and index_y < ring_y and index_y < pinky_y:
        return True
    return False

# def is_thumb_close_to_index(hand, thr=0.6) -> bool:
#     p1 = np.array([hand[fingers["wrist"][0]].x, hand[fingers["wrist"][0]].y])
#     p2 = np.array([hand[fingers["middle"][0]].x, hand[fingers["middle"][0]].y])
#
#     palm_size = distance(p1, p2)  # wrist to middle finger mcp
#
#     q1 = np.array([hand[fingers["thumb"][3]].x, hand[fingers["thumb"][3]].y])
#     q2 = np.array([hand[fingers["middle"][2]].x, hand[fingers["middle"][2]].y])
#
#     thumb_middle = distance(q1, q2)
#
#     print(f"Palm size: {palm_size:.3f}, Thumb-Index distance: {thumb_middle:.3f}")
#
#     if thumb_middle < palm_size * thr:
#         return True
#     return False


def change_color(hand) -> bool:
    return (is_finger_bent(hand, "index")
            and is_finger_bent(hand, "middle")
            and is_finger_bent(hand, "ring")
            and is_finger_bent(hand, "pinky")
            and is_finger_bent(hand, "thumb"))


# TODO + index tip higher than others (position)
# TODO + index far from wrist (distance)
def activate_drawing(hand) -> bool:
    return (is_finger_straight(hand, 'index')
            and is_index_highest(hand)
            and is_finger_bent(hand, 'middle')
            and is_finger_bent(hand, 'ring')
            and is_finger_bent(hand, 'pinky')
            and is_finger_bent(hand, 'thumb'))

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

def continue_drawing(frame, drawing_segments, current_segment):

    for segment in drawing_segments:
        if len(segment['points']) > 1:
            pts = np.array(segment['points'], dtype=np.int32)
            cv2.polylines(frame,
                          [pts],          # must be list of arrays
                          isClosed=False, # IMPORTANT for drawing strokes
                          color=segment['color'],
                          thickness=PEN_SIZE,
                          lineType=cv2.LINE_AA)

    if len(current_segment) > 1:
        pts = np.array(current_segment, dtype=np.int32)
        cv2.polylines(frame,
                      [pts],          # must be list of arrays
                      isClosed=False, # IMPORTANT for drawing strokes
                      color=PEN_COLOR,
                      thickness=PEN_SIZE,
                      lineType=cv2.LINE_AA)

def is_erasing(hand) -> bool:
    return (is_finger_bent(hand, 'index', thr=130)
            and is_finger_bent(hand, 'middle', thr=130)
            and is_finger_bent(hand, 'ring', thr=130)
            and is_finger_bent(hand, 'pinky', thr=130)
            and is_finger_bent(hand, 'thumb', thr=160))

# def erase_brush(segments, x, y, radius=5):
#     for segment in segments:
#         for i in range(len(segment)-1):
#             x1, y1 = segment[i]
#             x2, y2 = segment[i+1]
#
#             # Check if the line segment (x1, y1) to (x2, y2) is within the erasing radius
#             if (min(x1, x2) - radius <= x <= max(x1, x2) + radius and
#                 min(y1, y2) - radius <= y <= max(y1, y2) + radius):
#                 # If it is, remove the segment
#                 segment.remove((x1, y1))
#                 segment.remove((x2, y2))
#                 break
#
#                 def eraser_point(hand, frame_width, frame_height):
#                     # Average of all finger tips (thumb + index + middle + ring + pinky)
#                     points = [hand[4], hand[8], hand[12], hand[16], hand[20]]
#                     x = int(sum([p.x for p in points]) / len(points) * frame_width)
#                     y = int(sum([p.y for p in points]) / len(points) * frame_height)
#                     return (x, y)
#
#                 def erase_segments(segments, eraser_pos, radius=ERASER_RADIUS):
#                     # Remove segments that have a point within radius of eraser
#                     new_segments = []
#                     for segment, color in segments:
#                         erase = False
#                         for px, py in segment:
#                             if (px - eraser_pos[0]) ** 2 + (py - eraser_pos[1]) ** 2 <= radius ** 2:
#                                 erase = True
#                                 break
#                         if not erase:
#                             new_segments.append((segment, color))
#                     return new_segments

def eraser_center(hand, frame_width, frame_height):
    # Average of all finger tips (thumb + index + middle + ring + pinky)
    points = [hand[4], hand[8], hand[12], hand[16], hand[20]]
    x = int(sum([p.x for p in points]) / len(points) * frame_width)
    y = int(sum([p.y for p in points]) / len(points) * frame_height)
    return x, y

# Erase segments that have a point within radius of eraser
def erase_segments(segments, eraser_pos, radius=ERASER_RADIUS):
    radius_sq = radius ** 2
    ex, ey = eraser_pos

    return [
        seg for seg in segments
        if not any(
            (px - ex) ** 2 + (py - ey) ** 2 <= radius_sq
            for px, py in seg["points"]
        )
    ]

def draw_toolpad(img, pen_color, pen_size):
    """Draw UI ONLY for local preview."""
    overlay = img.copy()

    # panel
    cv2.rectangle(overlay, (20, 20), (260, 170), (0, 0, 0), -1)
    cv2.rectangle(overlay, (20, 20), (260, 170), (255, 255, 255), 2)

    cv2.putText(overlay, "AirBoard Toolpad", (30, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    # color preview
    cv2.putText(overlay, "Color:", (30, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2)
    cv2.rectangle(overlay, (120, 75), (240, 100), pen_color, -1)
    cv2.rectangle(overlay, (120, 75), (240, 100), (255,255,255), 1)

    # size preview
    cv2.putText(overlay, f"Size: {pen_size}", (30, 135),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2)
    cv2.circle(overlay, (200, 130), pen_size, (255,255,255), -1)

    cv2.imshow("AirBoard (local preview)", overlay)


