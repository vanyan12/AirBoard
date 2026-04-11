import cv2
import numpy as np

from config import DEFAULT_ERASER_RADIUS
from state import Point, Segment

fingers = {
    "thumb": [1, 2, 3, 4],
    "index": [5, 6, 7, 8],
    "middle": [9, 10, 11, 12],
    "ring": [13, 14, 15, 16],
    "pinky": [17, 18, 19, 20],
    "wrist": [0],
}

JOINT_INDEX = {
    "mcp": 0,
    "pip": 1,
    "dip": 2,
    "tip": 3,
}


def get_finger_joint(hand, finger_name: str, joint_name: str) -> np.ndarray:
    """
    Return a finger joint coordinate as np.array([x, y, z], dtype=np.float32).

    finger_name: "thumb" | "index" | "middle" | "ring" | "pinky"
    joint_name:  "mcp" | "pip" | "dip" | "tip"
    """
    joint_idx = fingers[finger_name][JOINT_INDEX[joint_name]]
    return np.array([hand[joint_idx].x, hand[joint_idx].y, hand[joint_idx].z], dtype=np.float32)


def angle(j1, j2, j3):
    v1 = j1 - j2
    v2 = j3 - j2
    cos_ang = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    angle_rad = np.arccos(cos_ang)
    angle_deg = np.degrees(angle_rad)
    return angle_deg


def distance(p1, p2):
    return np.linalg.norm(p1 - p2)


def is_finger_straight(hand, finger_name, thr=170) -> bool:
    mcp = get_finger_joint(hand, finger_name, "mcp")
    pip = get_finger_joint(hand, finger_name, "pip")
    dip = get_finger_joint(hand, finger_name, "dip")
    tip = get_finger_joint(hand, finger_name, "tip")

    return angle(mcp, pip, dip) > thr and angle(pip, dip, tip) > thr


def is_finger_extended(hand, finger_name, thr=150) -> bool:
    mcp = get_finger_joint(hand, finger_name, "mcp")
    pip = get_finger_joint(hand, finger_name, "pip")
    dip = get_finger_joint(hand, finger_name, "dip")
    tip = get_finger_joint(hand, finger_name, "tip")

    return angle(mcp, pip, dip) > thr and angle(pip, dip, tip) > thr


def is_finger_bent(hand, finger_name, thr=160) -> bool:
    mcp = get_finger_joint(hand, finger_name, "mcp")
    pip = get_finger_joint(hand, finger_name, "pip")
    dip = get_finger_joint(hand, finger_name, "dip")
    tip = get_finger_joint(hand, finger_name, "tip")

    if finger_name == "thumb":
        return angle(mcp, pip, dip) < 180 and angle(pip, dip, tip) < thr

    return angle(mcp, pip, dip) < thr and angle(pip, dip, tip) < thr


def smooth_point(new_x, new_y, prev_point, alpha=0.4) -> Point:
    if prev_point is None:
        return float(new_x), float(new_y)

    prev_x, prev_y = prev_point

    # speed = math.sqrt((new_x - prev_x)**2 + (new_y - prev_y)**2)
    # alpha = min(0.8, max(0.3, speed / 50))
    #
    # print(speed)


    smooth_x = alpha * new_x + (1 - alpha) * prev_x
    smooth_y = alpha * new_y + (1 - alpha) * prev_y
    return smooth_x, smooth_y


def is_index_highest(hand) -> bool:
    # y = 0 at top, y increases downward
    index_y = hand[8].y
    middle_y = hand[12].y
    ring_y = hand[16].y
    pinky_y = hand[20].y

    return (middle_y - index_y) >= 0.02 and (ring_y - index_y) >= 0.02 and (pinky_y - index_y) >= 0.02


def is_thumb_near_index(hand, thr=0.25) -> bool:
    thumb_dip = get_finger_joint(hand, "thumb", "dip")
    index_mcp = get_finger_joint(hand, "index", "mcp")

    print(distance(thumb_dip, index_mcp))
    return distance(thumb_dip, index_mcp) < thr


def are_pips_in_line(hand) -> bool:
    fingers_to_check = ["index", "middle", "ring", "pinky"]
    for finger_name in fingers_to_check:
        mcp = get_finger_joint(hand, finger_name, "mcp")
        pip = get_finger_joint(hand, finger_name, "pip")
        dip = get_finger_joint(hand, finger_name, "dip")
        tip = get_finger_joint(hand, finger_name, "tip")

        if angle(mcp, pip, dip) > 95 or angle(pip, dip, tip) > 95:
            return False
    return True

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
    return (
        is_finger_bent(hand, "index", thr=130)
        and is_finger_bent(hand, "middle")
        and is_finger_bent(hand, "ring")
        and is_finger_bent(hand, "pinky")
        # and is_finger_bent(hand, "thumb")
    )

def size_change(hand) -> bool:
    return (
        
    )

def activate_drawing(hand) -> bool:
    return (
        is_finger_extended(hand, "index")
        and is_index_highest(hand)
        # and not is_finger_extended(hand, "middle")
        # and not is_finger_extended(hand, "ring")
        # and not is_finger_extended(hand, "pinky")
        # and not is_finger_extended(hand, "thumb")
    )


def change_pen_color(hand) -> bool:
    return (
        activate_drawing(hand)
        and is_finger_straight(hand, "thumb")
        and is_thumb_near_index(hand, thr=0.3)
    )


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
        (0, 17), (13, 17), (17, 18), (18, 19), (19, 20),
    ]

    for connection in connections:
        start = hand_landmarks[connection[0]]
        end = hand_landmarks[connection[1]]

        x1, y1 = int(start.x * w), int(start.y * h)
        x2, y2 = int(end.x * w), int(end.y * h)

        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)


def draw_segments(frame, drawing_segments: list[Segment], current_segment: list[Point], current_color, current_size: int):
    for segment in drawing_segments:
        if len(segment.points) > 1:
            # Use np.round instead of direct cast for better rounding before int conversion
            pts = np.round(segment.points).astype(np.int32)
            cv2.polylines(
                frame,
                [pts],
                isClosed=False,
                color=segment.color,
                thickness=segment.size,
                lineType=cv2.LINE_AA,
            )

    if len(current_segment) > 1:
        # Use np.round instead of direct cast for better rounding before int conversion
        pts = np.round(current_segment).astype(np.int32)
        cv2.polylines(
            frame,
            [pts],
            isClosed=False,
            color=current_color,
            thickness=current_size,
            lineType=cv2.LINE_AA,
        )


def are_fingertips_clustered(hand, thr=0.25) -> bool:
    """Return True when all fingertips are close to each other."""
    tip_ids = [4, 8, 12, 16, 20]
    tips = [
        np.array([hand[i].x, hand[i].y, hand[i].z], dtype=np.float32)
        for i in tip_ids
    ]

    # All pairwise fingertip distances must be under the threshold.
    for i in range(len(tips)):
        for j in range(i + 1, len(tips)):
            if np.linalg.norm(tips[i] - tips[j]) > thr:
                return False

    return True


def is_erasing(hand) -> bool:
    return (
            is_finger_extended(hand, "index", thr=150)
            and is_finger_extended(hand, "middle", thr=150)
            and is_finger_extended(hand, "ring", thr=150)
            and is_finger_extended(hand, "pinky", thr=150)
            and is_finger_straight(hand, "thumb", thr=160)
            and are_fingertips_clustered(hand, thr=0.18)
    )

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
    points = [hand[5], hand[9], hand[13], hand[17], hand[0]]
    x = int(sum(p.x for p in points) / len(points) * frame_width)
    y = int(sum(p.y for p in points) / len(points) * frame_height)
    return x, y


def erase_segments(segments: list[Segment], eraser_pos, radius=DEFAULT_ERASER_RADIUS) -> list[Segment]:
    radius_sq = radius ** 2
    ex, ey = eraser_pos

    return [
        seg
        for seg in segments
        if not any((px - ex) ** 2 + (py - ey) ** 2 <= radius_sq for px, py in seg.points)
    ]
