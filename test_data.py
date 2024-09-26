import cv2
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 라벨과 각도를 불러옴
file = np.genfromtxt('data/merged_hand_angles.csv', delimiter=',')
angle = file[:, :-1].astype(np.float32)  # 각도 데이터
label = file[:, -1].astype(np.float32)   # 라벨 데이터
    
    
# KNN 모델 생성 및 학습 
knn = cv2.ml.KNearest_create()
knn.train(angle, cv2.ml.ROW_SAMPLE, label)

# 테스트 데이터 불러오기 (새로운 데이터로 테스트)
test_file = np.genfromtxt('data/merged_test_data.csv', delimiter=',')
test_angle = test_file[:, :-1].astype(np.float32)
test_label = test_file[:, -1].astype(np.float32)

# 테스트 데이터로 예측
ret, results, neighbours, dist = knn.findNearest(test_angle, 3)  # K=3
predictions = results.flatten()

# 정확도 계산
accuracy = accuracy_score(test_label, predictions)
print(f'Accuracy: {accuracy * 100:.2f}%')

# 혼동 행렬 계산
conf_matrix = confusion_matrix(test_label, predictions)

# 혼동 행렬 시각화
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=range(9), yticklabels=range(9))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title(f'Confusion Matrix (Accuracy: {accuracy * 100:.2f}%)')
plt.show()
