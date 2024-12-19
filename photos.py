import cv2
import mediapipe as mp
import numpy as np
import time

import keypress
import utils
import pyautogui

# Get the screen dimensions
screen_width, screen_height = pyautogui.size()

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
cap = cv2.VideoCapture(0)
window_width = screen_width // 3
window_height = screen_height // 3
cap.set(cv2.CAP_PROP_FRAME_WIDTH, window_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, window_height)

swipe_start = None
zoom_distance_start = None
gesture_timer = time.time()

detected_gesture = ''
gesture_display_time = 2
gesture_timestamp = 0
with mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8) as hands:

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        # Flip and convert the image
        image = cv2.flip(image, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process the image
        results = hands.process(image_rgb)

        # Draw the hand annotations
        image_height, image_width, _ = image.shape
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract features and call gesture detection functions
            # Add delay to prevent multiple detections
            if time.time() - gesture_timer > 1 :  
                lm_list = list()
                decoded_hand = dict()
                
                handedness = results.multi_handedness[0]
                wrist_z = hand_landmarks.landmark[0].z

                for lm in hand_landmarks.landmark:
                    cx = int(lm.x * image_width)
                    cy = int(lm.y * image_height)
                    cz = int((lm.z - wrist_z) * image_width)
                    lm_list.append([cx, cy, cz])
                
                label = handedness.classification[0].label.lower()
                lm_array = np.array(lm_list)
                direction, facing = utils.check_hand_direction(lm_array, label)
                boundary = utils.find_boundary_lm(lm_array)
                wrist_angle_joints = lm_array[[5, 0, 17]]
                wrist_angle = utils.calculate_angle(wrist_angle_joints)

                decoded_hand["label"] = label
                decoded_hand["landmarks"] = lm_array
                decoded_hand["wrist_angle"] = wrist_angle
                decoded_hand["direction"] = direction
                decoded_hand["facing"] = facing
                decoded_hand["boundary"] = boundary
                finger_states = utils.check_finger_states(decoded_hand)
                gestures = utils.get_gestures(label)
                gesture = utils.map_gesture(gestures, finger_states, decoded_hand["landmarks"], decoded_hand["wrist_angle"], decoded_hand["direction"], decoded_hand["boundary"])
                if gesture is not None:
                    print("Gesture Detected: ", gesture)
                    detected_gesture = gesture
                    gesture_timer = time.time()
                if gesture == "Thumbs-up":
                    keypress.mark_favorite()
                elif gesture == "Next Photo":
                    keypress.next_photo()
                elif gesture == "Previous Photo":
                    keypress.previous_photo()
                elif gesture == "Zoom In":
                    keypress.zoom_in()
                elif gesture == "Zoom Out":
                    keypress.zoom_out()
        if time.time() - gesture_timestamp > gesture_display_time:
            detected_gesture = ''

        cv2.imshow('Gesture Control', image)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()