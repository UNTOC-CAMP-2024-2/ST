import pandas as pd
import glob

# CSV 파일들의 경로 패턴 설정 (현재 디렉토리에 있는 csv 파일을 모두 읽어온다고 가정)
file_paths = glob.glob('data/*.csv')  # 모든 CSV 파일을 가져옴

# 모든 CSV 파일을 읽어서 리스트에 저장
dataframes = [pd.read_csv(file) for file in file_paths]

# 모든 DataFrame을 하나로 합치기
merged_df = pd.concat(dataframes, ignore_index=True)

# 합친 데이터프레임을 새로운 CSV 파일로 저장
merged_df.to_csv('data/merged_hand_angles.csv', index=False)

print("CSV 파일들이 성공적으로 합쳐졌습니다.")
