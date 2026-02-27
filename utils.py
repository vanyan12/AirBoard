import numpy as np


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
    'wrist': 0
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

def is_finger_straight(hand, finger_name, thr=165) -> bool:
    mcp = np.array([hand[fingers[finger_name][0]].x, hand[fingers[finger_name][0]].y, hand[fingers[finger_name][0]].z])
    pip = np.array([hand[fingers[finger_name][1]].x, hand[fingers[finger_name][1]].y, hand[fingers[finger_name][1]].z])
    dip = np.array([hand[fingers[finger_name][2]].x, hand[fingers[finger_name][2]].y, hand[fingers[finger_name][2]].z])
    tip = np.array([hand[fingers[finger_name][3]].x, hand[fingers[finger_name][3]].y, hand[fingers[finger_name][3]].z])

    angle_1 = angle(mcp, pip, dip)
    angle_2 = angle(pip, dip, tip)

    return angle_1 > thr and angle_2 > thr

def is_finger_bent(hand, finger_name) -> bool:

    mcp = np.array([hand[fingers[finger_name][0]].x, hand[fingers[finger_name][0]].y, hand[fingers[finger_name][0]].z])
    pip = np.array([hand[fingers[finger_name][1]].x, hand[fingers[finger_name][1]].y, hand[fingers[finger_name][1]].z])
    dip = np.array([hand[fingers[finger_name][2]].x, hand[fingers[finger_name][2]].y, hand[fingers[finger_name][2]].z])
    tip = np.array([hand[fingers[finger_name][3]].x, hand[fingers[finger_name][3]].y, hand[fingers[finger_name][3]].z])

    angle_1 = angle(mcp, pip, dip)
    angle_2 = angle(pip, dip, tip)

    return angle_1 < 165 or angle_2 < 165

def smooth_point(new_x, new_y, prev_point, alpha=0.4):
    if prev_point is None:
        return new_x, new_y

    prev_x, prev_y = prev_point

    smooth_x = int(alpha * new_x + (1 - alpha) * prev_x)
    smooth_y = int(alpha * new_y + (1 - alpha) * prev_y)

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
            and is_finger_bent(hand, 'middle')
            and is_finger_bent(hand, 'ring')
            and is_finger_bent(hand, 'pinky')
            and is_finger_bent(hand, 'thumb')
            and is_index_highest(hand))

def is_erasing(hand) -> bool:
    return (is_finger_straight(hand, 'index')
            and is_finger_straight(hand, 'middle')
            and is_finger_straight(hand, 'ring')
            and is_finger_straight(hand, 'pinky')
            and is_finger_straight(hand, 'thumb', thr=150))

def erase_brush(segments, x, y, radius=5):
    for segment in segments:
        for i in range(len(segment)-1):
            x1, y1 = segment[i]
            x2, y2 = segment[i+1]

            # Check if the line segment (x1, y1) to (x2, y2) is within the erasing radius
            if (min(x1, x2) - radius <= x <= max(x1, x2) + radius and
                min(y1, y2) - radius <= y <= max(y1, y2) + radius):
                # If it is, remove the segment
                segment.remove((x1, y1))
                segment.remove((x2, y2))
                break