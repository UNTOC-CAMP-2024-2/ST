import cv2 
import mediapipe as mp 
import numpy as np
###################
# from control_arduino import send_command
###################


"""
1. 주행
- 왼손 : 직진 or 후진 or 정지 결정 (한번 하면 상태 유지)
- 오른손 : 죄/우 회전 결정

2. 라이트
- 왼손 : 좌측 깜빡이, 우측 깜빡이
"""


"""
send message

100 전진 
101 전진 + 우회전
102 전진 + 좌회전

200 후진
201 후진 + 우회전
202 후진 + 좌회전

300 정지

400 비상깜빡이
401 헤드라이트 On
402 라이트 Off

500 클락션

"""





max_num_hands = 2 # 손 인식 개수
gesture = { 
    0:'go', 
    1:'back', 
    2:'stop', 
    3:'side',
    4:'head_light_on',
    5:'light_off'
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


# 상태 변수 초기화
left_hand_state = "stop"  # 왼손의 상태: "go", "back", "stop" 중 하나
left_hand_Light = None
right_hand_action = None  # 오른손의 동작: "left", "right", None 중 하나

motion_res = {
    'res':[],
    'cnt':0
}
while cap.isOpened(): # 웹캠에서 한 프레임씩 이미지를 읽어옴
    ret, img = cap.read()
    # left & right를 위한 초기화
    
    gesture[3] = 'side'
    if not ret:
        continue    

    img = cv2.flip(img, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    result = hands.process(img)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # 각도를 인식하고 제스처를 인식하는 부분 
    if result.multi_hand_landmarks is not None: 
        # 양손 모두 체크 
        for hand_idx, res in enumerate(result.multi_hand_landmarks):
            hand_label = result.multi_handedness[hand_idx].classification[0].label  # "Left" 또는 "Right"
            
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

            idx = int(results[0][0]) # 0, 1, 2, 3, 4
            
            # 모션 감지 여부
            if idx in gesture.keys():
                # 동일 모션 감지 여부 확인
                if len(motion_res["res"]) > 0 and motion_res["res"][-1] == idx:
                    motion_res["cnt"] += 1

                else:
                    # 새로운 모션이면 카운터 초기화
                    motion_res["cnt"] = 1  
                    
                motion_res["res"].append(idx) # 모션 추가
                
                # 동일 모션이 5번 초과이면 -> 동작 인식
                if motion_res["cnt"] > 5:
                    motion_res["res"] = [] # motion_res["res"] 초기화 
                    
                    # 왼손
                    if hand_label == "Left":
                        if gesture[idx] in ["go", "back", "stop"]:
                            left_hand_state = gesture[idx]  # 왼손 상태 저장
                           
                    # 오른손
                    elif hand_label == "Right":  
                        if gesture[idx] == "side":  # 좌/우 방향 결정
                            if res.landmark[4].x > res.landmark[8].x:
                                gesture[idx] = "left"
                                right_hand_action = "left"
                            elif res.landmark[4].x < res.landmark[8].x:
                                gesture[idx] = "right"
                                right_hand_action = "right"
                        elif gesture[idx] == "go": # 클락션
                            right_hand_action = "honk"
                        
                        elif gesture[idx] == "back":
                            right_hand_action = "emergency_ligth"
                            
                        elif gesture[idx] == "head_light_on":
                            right_hand_action = "head_light_on"
                        elif gesture[idx] == "light_off":
                            right_hand_action = "light_off"


                                
                    # 현재 상태를 바탕으로 행동 결정
                    if left_hand_state == "go":
                        if right_hand_action == "left":
                            # send_command(102) # go + left
                            print(102)
                        elif right_hand_action == "right":
                            # send_command(101) # go + right
                            print(101)
                        elif right_hand_action == "emergency_ligth":
                            # send_command(400) # go + emergency_ligth
                            print(400)
                        elif right_hand_action == "head_light_on":
                            # send_command(401) # go + head_light_on
                            print(401)
                        elif right_hand_action == "light_off":
                            # send_command(402) # go + light_off
                            print(402)
                        elif right_hand_action == "honk":
                            # send_command(500) # go + honk
                            print(500)
                        else:
                            # send_command(100) # just go
                            print(100)
                           
                    elif left_hand_state == "back":
                        if right_hand_action == "left":
                            # send_command(202)
                            print(202)
                        elif right_hand_action == "right":
                            # send_command(201)
                            print(201)
                        elif right_hand_action == "emergency_ligth":
                            # send_command(400) # go + emergency_ligth
                            print(400)
                        elif right_hand_action == "head_light_on":
                            # send_command(401) # go + head_light_on
                            print(401)
                        elif right_hand_action == "light_off":
                            # send_command(402) # go + light_off
                            print(402)
                        elif right_hand_action == "honk":
                            # send_command(500)
                            print(500)
                        else:
                            # send_command(200)
                            print(200)
                            
                    elif left_hand_state == "stop":
                        if right_hand_action == "emergency_ligth":
                            # send_command(400) # go + emergency_ligth
                            print(400)
                        elif right_hand_action == "head_light_on":
                            # send_command(401) # go + head_light_on
                            print(401)
                        elif right_hand_action == "light_off":
                            # send_command(402) # go + light_off
                            print(402)
                        elif right_hand_action == "honk":
                            # send_command(500)
                            print(500)
                        else:
                            # send_command(300)
                            print(300)
                                        
                        
                    right_hand_action = None
                    motion_res["cnt"] = 0 
                    
                    cv2.putText(img, text=gesture[idx].upper(), org=(int(res.landmark[0].x * img.shape[1]), int(res.landmark[0].y * img.shape[0] + 20)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
            mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS) # 손에 랜드마크를 그려줌 
        

    cv2.imshow('Game', img)
    if cv2.waitKey(1) == ord('q'):
        break

