import pandas as pd
import numpy as np
import torch
import pickle
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# 1. 데이터 불러오기
interactions_df = pd.read_csv('./data/travel_interactions.csv')

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

class TravelDataset(Dataset):
    def __init__(self, df, features, target):
        # 입력 데이터(X)와 정답 데이터(Y)를 텐서로 변환
        self.X = torch.tensor(df[features].values, dtype=torch.long)
        self.Y = torch.tensor(df[target].values, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

# DataLoader 생성 (배치 단위로 데이터 로드)
train_dataset = TravelDataset(train_df, features, target)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

import torch.nn as nn

class TravelRecommender(nn.Module):
    def __init__(self, num_categories, num_companions, num_places, embedding_dim=16):
        super(TravelRecommender, self).__init__()
        
        # 임베딩 레이어
        self.cat_embed = nn.Embedding(num_categories, embedding_dim)
        self.comp_embed = nn.Embedding(num_companions, embedding_dim)
        self.place_embed = nn.Embedding(num_places, embedding_dim)
        
        # 임베딩된 벡터들을 합친 후 통과할 완전 연결 계층 (MLP)
        self.fc_layers = nn.Sequential(
            nn.Linear(embedding_dim * 3, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1) # 최종 만족도 점수 출력
        )

    def forward(self, x):
        # x[:, 0]은 category, x[:, 1]은 companion_type, x[:, 2]는 place_name
        cat_vec = self.cat_embed(x[:, 0])
        comp_vec = self.comp_embed(x[:, 1])
        place_vec = self.place_embed(x[:, 2])
        
        # 세 벡터를 하나로 연결 (Concatenate)
        combined = torch.cat([cat_vec, comp_vec, place_vec], dim=1)
        
        # 점수 예측
        output = self.fc_layers(combined)
        return output.squeeze()

# 모델 초기화
num_categories = len(label_encoders['category'].classes_)
num_companions = len(label_encoders['companion_type'].classes_)
num_places = len(label_encoders['place_name'].classes_)

model = TravelRecommender(num_categories, num_companions, num_places)

import torch.optim as optim

criterion = nn.MSELoss() # 손실 함수
optimizer = optim.Adam(model.parameters(), lr=0.001) # 최적화 알고리즘

epochs = 10

for epoch in range(epochs):
    model.train()
    total_loss = 0
    
    for inputs, targets in train_loader:
        optimizer.zero_grad() # 기울기 초기화
        
        outputs = model(inputs) # 모델 예측
        loss = criterion(outputs, targets) # 오차 계산
        
        loss.backward() # 역전파
        optimizer.step() # 가중치 업데이트
        
        total_loss += loss.item()
        
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")

def recommend_places(selected_category, selected_companion, top_n=5):
    model.eval()
    
    # 1. 사용자 입력을 인코딩
    cat_idx = label_encoders['category'].transform([selected_category])[0]
    comp_idx = label_encoders['companion_type'].transform([selected_companion])[0]
    
    # 2. 평가할 모든 장소 목록 생성
    all_places_idx = np.arange(num_places)
    
    # 3. 모델 입력 텐서 생성 (모든 장소에 대해 동일한 사용자의 테마/동행을 조합)
    inputs = torch.tensor(
        [[cat_idx, comp_idx, p_idx] for p_idx in all_places_idx], 
        dtype=torch.long
    )
    
    # 4. 점수 예측
    with torch.no_grad():
        predicted_scores = model(inputs).numpy()
        
    # 5. 점수가 높은 순으로 정렬하여 상위 N개 추출
    top_indices = predicted_scores.argsort()[-top_n:][::-1]
    
    # 6. 인덱스를 다시 원래 장소 이름으로 디코딩
    recommended_places = label_encoders['place_name'].inverse_transform(top_indices)
    
    return recommended_places

# 예시 실행: "쇼핑" 테마로 "친구와 떠나는 여행"을 선택했을 때 추천받기
top_places = recommend_places("쇼핑", "친구와 떠나는 여행", top_n=5)
print("추천 장소:", top_places)

# 1. 모델 가중치 저장
torch.save(model.state_dict(), 'travel_recommender.pth')
print("모델 가중치가 'travel_recommender.pth'로 저장되었습니다.")

# 2. 인코더 저장 (나중에 웹에서 입력받은 텍스트를 다시 숫자로 바꾸기 위해 필요)
with open('label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)
print("인코더가 'label_encoders.pkl'로 저장되었습니다.")