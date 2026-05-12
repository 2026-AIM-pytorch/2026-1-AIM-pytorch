import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# 1. 데이터 불러오기
interactions_df = pd.read_csv('travel_interactions.csv')

# 2. 사용할 특성(Feature)과 타겟(Target) 설정
# 사용자가 선택하는 'category(테마)', 'companion_type(동행)'과 'place_name(장소)'를 주요 특성으로 사용
features = ['category', 'companion_type', 'place_name']
target = 'schedule_satisfaction' 

# 3. 범주형 데이터 인코딩 (문자열 -> 숫자)
label_encoders = {}
for col in features:
    le = LabelEncoder()
    interactions_df[col] = le.fit_transform(interactions_df[col])
    label_encoders[col] = le # 나중에 추론할 때 역변환을 위해 저장

# 4. 학습/테스트 데이터 분리
train_df, test_df = train_test_split(interactions_df, test_size=0.2, random_state=42)