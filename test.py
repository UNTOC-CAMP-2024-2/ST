import cv2 
import mediapipe as mp 
import numpy as np

# MediaPipe hands model
mp_hands = mp.solutions.hands 
mp_drawing = mp.solutions.drawing_utils

def initialize_hand_module(max_num_hands=2):
    return mp_hands.Hands(
        max_num_hands=max_num_hands,
        min_detection_confidence=0.5, 
        min_tracking_confidence=0.5
    )

def initialize_knn_model(file_path):
    file = np.genfromtxt(file_path, delimiter=',')
    angle = file[:, :-1].astype(np.float32)
    label = file[:, -1].astype(np.float32)
    knn = cv2.ml.KNearest_create()
    knn.train(angle, cv2.ml.ROW_SAMPLE, label)
    return knn

def detect_gesture(hand_landmarks, knn):
    joint = np.zeros((21, 3))
    for j, lm in enumerate(hand_landmarks):
        joint[j] = [lm.x, lm.y, lm.z]

    v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :]
    v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :]
    v = v2 - v1
    v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

    angle = np.arccos(np.einsum('nt,nt->n',
                                v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18], :],
                                v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19], :]))
    angle = np.degrees(angle)

    data = np.array([angle], dtype=np.float32)
    ret, results, neighbours, dist = knn.findNearest(data, 3)
    return int(results[0][0])

def process_hand_gestures(results, knn, gesture, motion_res, left_hand_state, right_hand_action):
    if results.multi_hand_landmarks is None:
        return left_hand_state, right_hand_action

    for hand_idx, res in enumerate(results.multi_hand_landmarks):
        hand_label = results.multi_handedness[hand_idx].classification[0].label
        idx = detect_gesture(res.landmark, knn)

        if idx in gesture:
            if motion_res['res'] and motion_res['res'][-1] == idx:
                motion_res['cnt'] += 1
            else:
                motion_res['cnt'] = 1

            motion_res['res'].append(idx)

            if motion_res['cnt'] > 5:
                motion_res['res'] = []
                motion_res['cnt'] = 0
                
                if hand_label == 'Left' and gesture[idx] in ["go", "back", "stop"]:
                    left_hand_state = gesture[idx]

                elif hand_label == 'Right':
                    if gesture[idx] == 'side':
                        if res.landmark[4].x > res.landmark[8].x:
                            right_hand_action = "left"
                        elif res.landmark[4].x < res.landmark[8].x:
                            right_hand_action = "right"
                    elif gesture[idx] in ["go", "back", "head_light_on", "light_off"]:
                        right_hand_action = gesture[idx]

    return left_hand_state, right_hand_action

def determine_action(left_hand_state, right_hand_action):
    if left_hand_state == "go":
        return {
            "left": 102, "right": 101, "emergency_light": 400, "head_light_on": 401,
            "light_off": 402, "honk": 500
        }.get(right_hand_action, 100)

    if left_hand_state == "back":
        return {
            "left": 202, "right": 201, "emergency_light": 400, "head_light_on": 401,
            "light_off": 402, "honk": 500
        }.get(right_hand_action, 200)

    if left_hand_state == "stop":
        return {
            "emergency_light": 400, "head_light_on": 401, "light_off": 402, "honk": 500
        }.get(right_hand_action, 300)

    return None

def main():
    max_num_hands = 2
    gesture = {0: 'go', 1: 'back', 2: 'stop', 3: 'side', 4: 'head_light_on', 5: 'light_off'}

    hands = initialize_hand_module(max_num_hands)
    knn = initialize_knn_model('training/merged_training.csv')

    cap = cv2.VideoCapture(0)

    left_hand_state = "stop"
    right_hand_action = None
    motion_res = {'res': [], 'cnt': 0}

    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            continue

        img = cv2.flip(img, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        left_hand_state, right_hand_action = process_hand_gestures(
            results, knn, gesture, motion_res, left_hand_state, right_hand_action
        )

        action = determine_action(left_hand_state, right_hand_action)
        if action:
            print(action)
            right_hand_action = None

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.imshow('Game', img)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
