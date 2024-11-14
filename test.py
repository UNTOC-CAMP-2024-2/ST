import cv2 
import mediapipe as mp 
import numpy as np
import time
max_num_hands = 2 # 손 인식 개수
gesture = { 
    0:'0', 
    1:'1', 
    2:'2', 
    3:'3', 
    4:'4', 
    5:'5',
    6:"good",
    7:"ok",
    8:"peace"
    }

# MediaPipe hands model
mp_hands = mp.solutions.hands 
mp_drawing = mp.solutions.drawing_utils

# 손가락 detection 모듈을 초기화
hands = mp_hands.Hands(  
    max_num_hands=max_num_hands,
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5)  



# 제스처 인식 모델 
file = np.genfromtxt('training/merged_training.csv', delimiter=',')
angle = file[:,:-1].astype(np.float32) # 각도
label = file[:, -1].astype(np.float32) # 라벨
knn = cv2.ml.KNearest_create() # knn(k-최근접 알고리즘)
knn.train(angle, cv2.ml.ROW_SAMPLE, label) # 학습

cap = cv2.VideoCapture(0) 

while cap.isOpened(): # 웹캠에서 한 프레임씩 이미지를 읽어옴
    ret, img = cap.read()
    if not ret:
        continue    

    img = cv2.flip(img, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    hand_process = hands.process(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # 오른손 or 왼손
    hand_side = []
    
    # 각도를 인식하고 제스처를 인식하는 부분 
    if hand_process.multi_hand_landmarks is not None: 
        for res in hand_process.multi_hand_landmarks: 
            print(res)
            joint = np.zeros((21, 3)) 
            for j, lm in enumerate(res.landmark):
                joint[j] = [lm.x, lm.y, lm.z] # 각 joint마다 x,y,z 좌표 저장

            v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19],:]
            v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],:]
            v = v2 - v1  
            
            # 뼈의 길이 벡터
            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis] # 벡터 정규화(크기 1 벡터) = v / 벡터의 크기

            # 손의 움직임을 구성하는 주요 각도 15개의 각도를 계산 
            angle = np.arccos(np.einsum('nt,nt->n',
                v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
                v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:]))

            angle = np.degrees(angle) # radian 각도를 degree 각도로 변환

            data = np.array([angle], dtype=np.float32)
            ret, results, neighbours, dist = knn.findNearest(data, 3) # k가 3일 때 값을 구한다

            idx = int(results[0][0]) 

            # 오른손 : 1 / 왼손 : 0
            # if hand_process.multi_handedness == 1:
            #     hand = 1
            # elif hand_process.multi_handedness == 0:
            #     hand = 0
                
                
            # for i, hand in enumerate(hand_process.multi_handedness):
            #     hand_side.append(hand[i].classification[0].label)
                
                
            # print(hand_process.multi_handedness[0].classification[0].label)   

            # for i, hand in enumerate(hand_process.multi_handedness):
            #     hand_side.append(hand.classification[0].label)
            #     print(i, hand_side)
    
            
            if idx in gesture.keys(): 
                cv2.putText(img, text=f"{gesture[idx].upper()}", org=(int(res.landmark[0].x * img.shape[1]), int(res.landmark[0].y * img.shape[0] + 20)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)

            mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS) # 손에 랜드마크를 그려줌 
            
    cv2.imshow('Game', img)
    if cv2.waitKey(1) == ord('q'):
        break

