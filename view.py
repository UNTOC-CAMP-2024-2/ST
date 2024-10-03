import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# CSV 파일을 읽기
df = pd.read_csv('training/merged_training.csv', header=None)

# 데이터와 라벨 분리 (마지막 열이 라벨)
features = df.iloc[:, :-1]  # 마지막 열 제외한 나머지는 feature
labels = df.iloc[:, -1]     # 마지막 열이 label

# PCA로 차원 축소 (2차원으로)
pca = PCA(n_components=2)
reduced_features = pca.fit_transform(features)

# 8개의 라벨을 위한 색상 맵핑 (0부터 7까지)
colors = plt.cm.get_cmap('tab10', 8)

# 2차원으로 축소된 데이터 시각화
plt.figure(figsize=(10, 8))
scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=labels, cmap=colors, alpha=0.7)

# 컬러바 추가
cbar = plt.colorbar(scatter, ticks=range(8))
cbar.set_label('Label')

# 축 라벨과 제목 추가
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('PCA of Features (Reduced to 2D) Color-Coded by Label')

# 시각화 표시
plt.show()
