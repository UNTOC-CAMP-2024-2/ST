import cv2 as cv
import mediapipe as mp
import numpy as np
import time
import pandas as pd

# MediaPipe 및 Hand Tracking 설정
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

# 동영상 파일 불러오기
cap = cv.VideoCapture(0)  # 웹캠 사용

# 시간 간격 조정을 위한 변수
prev_time = 0
interval = 0.5  # 1초 간격

# 각도 데이터를 저장할 리스트
angles_list = []

test_data = 0
test_case = {
    0 : "0",
    1 : "1",
    2 : "2",
    3 : "3",
    4 : "4",
    5 : "5",
    6 : "good",
    7 : "ok",
    8 : "peace"
}
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 현재 시간 체크
    current_time = time.time()

    # BGR을 RGB로 변환 (MediaPipe는 RGB 이미지를 기대함)
    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    # MediaPipe로 손 검출
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            joint = np.zeros((21, 3))

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            for i, landmark in enumerate(hand_landmarks.landmark):
                joint[i] = [landmark.x, landmark.y, landmark.z]

            v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :]
            v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :]
            # 뼈의 길이
            v = v2 - v1
            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

            # 뼈의 값으로 뼈사이의 각도 구하기, 변화값이 큰 15개
            angle = np.arccos(np.einsum('nt,nt->n',
                                        v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],
                                        v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]))

            # radian 각도를 degree각도로 변경하기
            angle = np.degrees(angle)

            # 1초마다 각도 출력 및 저장
            if current_time - prev_time >= interval:
                print(f"angle: {angle}")
                extended_angle = np.append(angle, test_data)
                angles_list.append(extended_angle)  # 수정된 각도를 리스트에 저장
                prev_time = current_time

    cv.imshow('Hand Tracking', frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# 비디오 캡처 해제
cap.release()
cv.destroyAllWindows()

# 각도 데이터를 DataFrame으로 변환
df = pd.DataFrame(angles_list)

# csv 파일로 저장
# df.to_csv(f'data/hand_angles_{test_data}.csv', index=False)
df.to_csv(f'data/test_data.csv', index=False)
