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

def is_finger_straight(hand, finger_name) -> bool:
    mcp = np.array([hand[fingers[finger_name][0]].x, hand[fingers[finger_name][0]].y, hand[fingers[finger_name][0]].z])
    pip = np.array([hand[fingers[finger_name][1]].x, hand[fingers[finger_name][1]].y, hand[fingers[finger_name][1]].z])
    dip = np.array([hand[fingers[finger_name][2]].x, hand[fingers[finger_name][2]].y, hand[fingers[finger_name][2]].z])
    tip = np.array([hand[fingers[finger_name][3]].x, hand[fingers[finger_name][3]].y, hand[fingers[finger_name][3]].z])

    angle_1 = angle(mcp, pip, dip)
    angle_2 = angle(pip, dip, tip)

    return angle_1 > 165 and angle_2 > 165

def is_finger_bent(hand, finger_name) -> bool:

    mcp = np.array([hand[fingers[finger_name][0]].x, hand[fingers[finger_name][0]].y, hand[fingers[finger_name][0]].z])
    pip = np.array([hand[fingers[finger_name][1]].x, hand[fingers[finger_name][1]].y, hand[fingers[finger_name][1]].z])
    dip = np.array([hand[fingers[finger_name][2]].x, hand[fingers[finger_name][2]].y, hand[fingers[finger_name][2]].z])
    tip = np.array([hand[fingers[finger_name][3]].x, hand[fingers[finger_name][3]].y, hand[fingers[finger_name][3]].z])

    angle_1 = angle(mcp, pip, dip)
    angle_2 = angle(pip, dip, tip)

    return angle_1 < 165 or angle_2 < 165

# TODO + index tip higher than others (position)
# TODO + index far from wrist (distance)
# TODO + gesture stable for 5 frames
def activate_drawing(hand) -> bool:
    return (is_finger_straight(hand, 'index')
            and is_finger_bent(hand, 'middle')
            and is_finger_bent(hand, 'ring')
            and is_finger_bent(hand, 'pinky')
            and is_finger_bent(hand, 'thumb'))


#Erasing function: all fingers straight except pinky 


