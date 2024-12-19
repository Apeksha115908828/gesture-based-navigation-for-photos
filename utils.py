import numpy as np
import mediapipe as mp
mp_hands = mp.solutions.hands
swipe_start = None

zoom_distance_start = None
BENT_RATIO_THRESH = [0.76, 0.88, 0.85, 0.65]
THUMB_THRESH = [9, 8]
NON_THUMB_THRESH = [8.6, 7.6, 6.6, 6.1]
mcp_joints = [5, 9, 13, 17]

def check_hand_direction(landmarks, label):
    direction = None
    facing = None
    
    wrist = landmarks[0]
    thumb_mcp = landmarks[1]
    pinky_mcp = landmarks[17]

    mcp_x_avg = np.mean(landmarks[mcp_joints, 0])
    mcp_y_avg = np.mean(landmarks[mcp_joints, 1])

    mcp_wrist_x = np.absolute(mcp_x_avg - wrist[0])
    mcp_wrist_y = np.absolute(mcp_y_avg - wrist[1])

    if mcp_wrist_x > mcp_wrist_y:
        if mcp_x_avg < wrist[0]:
            direction = 'left'
            if label == 'left':
                facing = 'front' if thumb_mcp[1] < pinky_mcp[1] else 'back'
            else:
                facing = 'front' if thumb_mcp[1] > pinky_mcp[1] else 'back'
        else:
            direction = 'right'
            if label == 'left':
                facing = 'front' if thumb_mcp[1] > pinky_mcp[1] else 'back'
            else:
                facing = 'front' if thumb_mcp[1] < pinky_mcp[1] else 'back'
    else:
        if mcp_y_avg < wrist[1]:
            direction = 'up'
            if label == 'left':
                facing = 'front' if thumb_mcp[0] > pinky_mcp[0] else 'back'
            else:
                facing = 'front' if thumb_mcp[0] < pinky_mcp[0] else 'back'
        else:
            direction = 'down'
            if label == 'left':
                facing = 'front' if thumb_mcp[0] < pinky_mcp[0] else 'back'
            else:
                facing = 'front' if thumb_mcp[0] > pinky_mcp[0] else 'back'
    
    return direction, facing

def map_gesture(gestures, finger_states, landmarks, wrist_angle, direction, boundary):    
    detected_gesture = None
    d = two_landmark_distance(landmarks[0], landmarks[5])
    thresh = d / 4
    for ges, temp in gestures.items():
        count = 0
        flag = 0
        for i in range(len(finger_states)):
            if finger_states[i] not in temp['finger states'][i]:
                flag = 1
                break
        if flag == 0:
            count += 1
        if temp['wrist angle'] is None:
            count += 1
        elif temp['wrist angle'][0] < wrist_angle < temp['wrist angle'][1]:
            count += 1
        if temp['direction'] == direction:
            count += 1
        if temp['overlap'] is None:
            count += 1
        else:
            flag = 0
            for lm1, lm2 in temp['overlap']:
                if two_landmark_distance(landmarks[lm1], landmarks[lm2]) > thresh:
                    flag = 1
                    break
            if flag == 0:
                count += 1
        if temp['boundary'] is None:
            count += 1
        else:
            flag = 0
            for bound, lm in temp['boundary'].items():
                if boundary[bound] not in lm:
                    flag = 1
                    break
            if flag == 0:
                count += 1
        
        if count == 5:
            detected_gesture = ges
            break
    if detected_gesture is None :
        gesture = detect_swipe(landmarks)
        if gesture is not None:
            return gesture
    if detected_gesture is None :
        gesture = detect_zoom(landmarks)
        if gesture is not None:
            return gesture

    return detected_gesture

def get_finger_state(joint_angles, threshold):
    acc_angle = joint_angles.sum()
    finger_state = None
    
    new_threshold = threshold.copy()
    new_threshold.append(-np.inf)
    new_threshold.insert(0, np.inf)
    
    for i in range(len(new_threshold)-1):
        if new_threshold[i] > acc_angle >= new_threshold[i+1]:
            finger_state = i
            break
    
    return finger_state

def calculate_thumb_angle(joints, label, facing):
    vec1 = joints[0][:2] - joints[1][:2]
    vec2 = joints[2][:2] - joints[1][:2]

    if label == 'left':
        cross = np.cross(vec1, vec2) if facing == 'front' else np.cross(vec2, vec1)
    else:
        cross = np.cross(vec2, vec1) if facing == 'front' else np.cross(vec1, vec2)
    dot = np.dot(vec1, vec2)
    angle = np.arctan2(cross, dot)
    if angle < 0:
        angle += 2 * np.pi
    
    return angle

def two_landmark_distance(vec1, vec2, dim=2):
    vec = vec2[:dim] - vec1[:dim]
    distance = np.linalg.norm(vec)
    
    return distance

def get_gestures(label):
    return {
    'Thumbs-up':    {
        'finger states':   [[0], [3, 4], [3, 4], [3, 4], [3, 4]],
        'direction':       'left' if label == 'right' else 'right',
        'wrist angle':     [0.80, 1.50],
        'overlap':         None,
        'boundary':        {3: [4]}},
    }

def detect_zoom(landmarks):
    global zoom_distance_start

    index_finger_tip = np.array(landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP.value])
    thumb_tip = np.array(landmarks[mp_hands.HandLandmark.THUMB_TIP.value])

    distance = np.linalg.norm(index_finger_tip - thumb_tip)
    if zoom_distance_start is None:
        zoom_distance_start = distance
        return None
    else:
        zoom_change = distance - zoom_distance_start

        if zoom_change > 30:
            print("Zoom In Gesture Detected")
            zoom_distance_start = None
            return "Zoom In"
        elif zoom_change < -30:
            print("Zoom Out Gesture Detected")
            zoom_distance_start = None
            return "Zoom Out"
        return None

def detect_swipe(landmarks):
    global swipe_start
    index_finger_tip = np.array(landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP.value])
    if swipe_start is None:
        swipe_start = index_finger_tip
        return None
    else:
        swipe_end = index_finger_tip
        swipe_vector = np.array(swipe_end) - np.array(swipe_start)

        if np.linalg.norm(swipe_vector) > 100:
            angle = np.arctan2(swipe_vector[1], swipe_vector[0]) * 180 / np.pi

            if -45 < angle < 45:
                print("Swipe Right to Left Detected: Next Photo")
                return "Next Photo"
            elif 135 < angle or angle < -135:
                print("Swipe Left to Right Detected: Previous Photo")
                return "Previous Photo"
            swipe_start = None
        return None
    
def find_boundary_lm(landmarks):
    xs = landmarks[:,0]
    ys = landmarks[:,1]
    lm_x_max, lm_x_min = np.argmax(xs), np.argmin(xs)
    lm_y_max, lm_y_min = np.argmax(ys), np.argmin(ys)

    return [lm_x_max, lm_x_min, lm_y_max, lm_y_min]

def calculate_angle(joints):
    vec1 = joints[0][:2] - joints[1][:2]
    vec2 = joints[2][:2] - joints[1][:2]

    cross = np.cross(vec1, vec2)
    dot = np.dot(vec1, vec2)
    angle = np.absolute(np.arctan2(cross, dot))

    return angle

def check_finger_states(hand):
    landmarks = hand['landmarks']
    label = hand['label']
    facing = hand['facing']

    finger_states = [None] * 5
    joint_angles = np.zeros((5,3))

    d1 = two_landmark_distance(landmarks[0], landmarks[5])
    
    for i in range(5):
        joints = [0, 4*i+1, 4*i+2, 4*i+3, 4*i+4]
        if i == 0:
            joint_angles[i] = np.array(
                [calculate_thumb_angle(landmarks[joints[j:j+3]], label, facing) for j in range(3)]
            )
            finger_states[i] = get_finger_state(joint_angles[i], THUMB_THRESH)
        else:
            joint_angles[i] = np.array(
                [calculate_angle(landmarks[joints[j:j+3]]) for j in range(3)]
            )
            d2 = two_landmark_distance(landmarks[joints[1]], landmarks[joints[4]])
            finger_states[i] = get_finger_state(joint_angles[i], NON_THUMB_THRESH)
            
            if finger_states[i] == 0 and d2/d1 < BENT_RATIO_THRESH[i-1]:
                finger_states[i] = 1
    
    return finger_states