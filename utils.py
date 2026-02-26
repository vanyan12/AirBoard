import numpy as np

# Finger landmark indexes (MediaPipe)
fingers = {
    'thumb': [1, 2, 3, 4],
    'index': [5, 6, 7, 8],
    'middle': [9, 10, 11, 12],
    'ring': [13, 14, 15, 16],
    'pinky': [17, 18, 19, 20],
    'wrist': 0
}

def angle(j1, j2, j3):
    v1 = j1 - j2
    v2 = j3 - j2

    norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norm_product == 0:
        return 0

    cos_ang = np.dot(v1, v2) / norm_product
    cos_ang = np.clip(cos_ang, -1.0, 1.0)

    angle_rad = np.arccos(cos_ang)
    return np.degrees(angle_rad)

def is_finger_straight(hand, finger_name):
    mcp = np.array([hand[fingers[finger_name][0]].x,
                    hand[fingers[finger_name][0]].y,
                    hand[fingers[finger_name][0]].z])

    pip = np.array([hand[fingers[finger_name][1]].x,
                    hand[fingers[finger_name][1]].y,
                    hand[fingers[finger_name][1]].z])

    dip = np.array([hand[fingers[finger_name][2]].x,
                    hand[fingers[finger_name][2]].y,
                    hand[fingers[finger_name][2]].z])

    tip = np.array([hand[fingers[finger_name][3]].x,
                    hand[fingers[finger_name][3]].y,
                    hand[fingers[finger_name][3]].z])

    angle_1 = angle(mcp, pip, dip)
    angle_2 = angle(pip, dip, tip)

    return angle_1 > 150 and angle_2 > 150

def is_finger_bent(hand, finger_name):
    mcp = np.array([hand[fingers[finger_name][0]].x,
                    hand[fingers[finger_name][0]].y,
                    hand[fingers[finger_name][0]].z])

    pip = np.array([hand[fingers[finger_name][1]].x,
                    hand[fingers[finger_name][1]].y,
                    hand[fingers[finger_name][1]].z])

    dip = np.array([hand[fingers[finger_name][2]].x,
                    hand[fingers[finger_name][2]].y,
                    hand[fingers[finger_name][2]].z])

    tip = np.array([hand[fingers[finger_name][3]].x,
                    hand[fingers[finger_name][3]].y,
                    hand[fingers[finger_name][3]].z])

    angle_1 = angle(mcp, pip, dip)
    angle_2 = angle(pip, dip, tip)

    return angle_1 < 150 or angle_2 < 150


# GREEN MODE → Only index up
def activate_drawing(hand):
    return (is_finger_straight(hand, 'index')
            and is_finger_bent(hand, 'middle')
            and is_finger_bent(hand, 'ring')
            and is_finger_bent(hand, 'pinky'))


# RED MODE → Only pinky up
def activate_red_drawing(hand):
    return (is_finger_straight(hand, 'pinky')
            and is_finger_bent(hand, 'index')
            and is_finger_bent(hand, 'middle')
            and is_finger_bent(hand, 'ring'))

# utils.py

def activate_eraser(hand):
    return (is_finger_bent(hand, 'index') and
            is_finger_bent(hand, 'middle') and
            is_finger_bent(hand, 'ring') and
            is_finger_bent(hand, 'pinky'))
