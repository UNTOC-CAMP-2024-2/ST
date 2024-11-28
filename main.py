import cv2 
import mediapipe as mp 
import numpy as np
###################
# from control_arduino import send_command
###################


max_num_hands = 1 # 손 인식 개수
gesture = { 
    0:'go', 
    1:'back', 
    2:'stop', 
    3:'side',
    4:'backLight'
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
motion_res = {
    'res':[],
    'cnt':0
}
while cap.isOpened(): # 웹캠에서 한 프레임씩 이미지를 읽어옴
    ret, img = cap.read()
    # left & right를 위한 초기화
    gesture[3] = 'side'
    gesture[4] = "backLight"
    if not ret:
        continue    

    img = cv2.flip(img, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    result = hands.process(img)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # 각도를 인식하고 제스처를 인식하는 부분 
    if result.multi_hand_landmarks is not None: 
        for res in result.multi_hand_landmarks: 
            joint = np.zeros((21, 3)) 
            # print() 
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
            if idx in gesture.keys():
                # 동일 모션 감지 여부 확인
                if len(motion_res["res"]) > 0 and motion_res["res"][-1] == idx:
                    motion_res["cnt"] += 1
                else:
                    motion_res["cnt"] = 1  # 새로운 모션이면 카운터 초기화
                
                motion_res["res"].append(idx)
                
                if motion_res["cnt"] > 5:
                    # left & right
                    if gesture[idx] == "side" and res.landmark[4].x > res.landmark[8].x:
                        gesture[idx] = "left"
                    elif gesture[idx] == "side" and res.landmark[4].x < res.landmark[8].x:
                        gesture[idx] = "right"
                        
                    if gesture[idx] == "backLight" and res.landmark[17].x > res.landmark[4].x:
                        print("left")
                        gesture[idx] = "leftLight"
                    elif gesture[idx] == "backLight" and res.landmark[17].x < res.landmark[4].x:
                        print("right")
                        gesture[idx] = "rightLight"
                        
                    motion_res["cnt"] = 0 
                    cv2.putText(img, text=gesture[idx].upper(), org=(int(res.landmark[0].x * img.shape[1]), int(res.landmark[0].y * img.shape[0] + 20)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
                # send_command(str(idx))
            mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS) # 손에 랜드마크를 그려줌 

    cv2.imshow('Game', img)
    if cv2.waitKey(1) == ord('q'):
        break

