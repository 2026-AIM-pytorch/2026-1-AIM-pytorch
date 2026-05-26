import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import pickle

# ─────────────────────────────────────────────
# 0. 재현성 고정
# ─────────────────────────────────────────────
torch.manual_seed(42)
np.random.seed(42)

# ─────────────────────────────────────────────
# 1. 데이터 로드
# ─────────────────────────────────────────────
df = pd.read_csv('travel_interactions.csv')

# 기존 코드에 없던 특성 추가 (데이터 기반)
# revisit(Y/N), crowdedness, stay_time_minutes, mobility_preference 활용
cat_features   = ['category', 'companion_type', 'place_name', 'mood',
                  'district', 'crowdedness', 'mobility_preference', 'revisit']
num_features   = ['age', 'google_rating', 'stay_time_minutes']   # stay_time 추가
target         = 'schedule_satisfaction'

# ─────────────────────────────────────────────
# 2. 전처리 (데이터 누수 방지: split → fit)
# ─────────────────────────────────────────────
# NaN 방어 처리
for col in cat_features:
    df[col] = df[col].fillna('unknown').astype(str)

# 먼저 분리
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# 범주형: 전체 데이터로 fit (배포 환경에서 미등장 클래스 방지)
label_encoders = {}
for col in cat_features:
    le = LabelEncoder()
    le.fit(df[col])                                     # 전체 기준 fit
    train_df = train_df.copy()
    test_df  = test_df.copy()
    train_df[col] = le.transform(train_df[col])
    test_df[col]  = le.transform(test_df[col])
    label_encoders[col] = le

# 수치형: train으로만 fit → test에 transform (누수 차단)
scaler = StandardScaler()
train_df[num_features] = scaler.fit_transform(train_df[num_features])
test_df[num_features]  = scaler.transform(test_df[num_features])

# target 스케일링 (62~100 범위 → 정규화)
target_scaler = StandardScaler()
train_df[[target]] = target_scaler.fit_transform(train_df[[target]])
test_df[[target]]  = target_scaler.transform(test_df[[target]])

# ─────────────────────────────────────────────
# 3. Dataset
# ─────────────────────────────────────────────
class TravelDataset(Dataset):
    def __init__(self, df, cat_cols, num_cols, target_col):
        self.X_cat = torch.tensor(df[cat_cols].values, dtype=torch.long)
        self.X_num = torch.tensor(df[num_cols].values, dtype=torch.float32)
        self.Y     = torch.tensor(df[target_col].values, dtype=torch.float32)

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        return self.X_cat[idx], self.X_num[idx], self.Y[idx]

train_ds     = TravelDataset(train_df, cat_features, num_features, target)
test_ds      = TravelDataset(test_df,  cat_features, num_features, target)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=64, shuffle=False)   # 평가용 shuffle=False

# ─────────────────────────────────────────────
# 4. 모델 (어휘 크기 비례 임베딩 차원 적용)
# ─────────────────────────────────────────────
class AdvancedRecommender(nn.Module):
    def __init__(self, vocab_sizes, num_numeric):
        super().__init__()

        # 어휘 크기에 비례한 임베딩 차원 (rule of thumb: min(50, (n+1)//2))
        self.embeddings = nn.ModuleDict({
            col: nn.Embedding(size, min(50, (size + 1) // 2))
            for col, size in vocab_sizes.items()
        })

        embed_total = sum(min(50, (size + 1) // 2) for size in vocab_sizes.values())
        input_dim   = embed_total + num_numeric

        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x_cat, x_num):
        embedded = [
            self.embeddings[col](x_cat[:, i])
            for i, col in enumerate(self.embeddings.keys())
        ]
        x = torch.cat(embedded + [x_num], dim=1)
        return self.network(x).squeeze()

# ─────────────────────────────────────────────
# 5. 초기화
# ─────────────────────────────────────────────
vocab_sizes = {col: len(label_encoders[col].classes_) for col in cat_features}
model       = AdvancedRecommender(vocab_sizes, len(num_features))
optimizer   = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler   = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
criterion   = nn.MSELoss()

# ─────────────────────────────────────────────
# 6. 학습 루프
# ─────────────────────────────────────────────
NUM_EPOCHS  = 30
best_val_loss = float('inf')

for epoch in range(NUM_EPOCHS):
    # --- train ---
    model.train()
    train_loss = 0.0
    for x_cat, x_num, y in train_loader:
        optimizer.zero_grad()
        pred = model(x_cat, x_num)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)

    # --- validation ---
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for x_cat, x_num, y in test_loader:
            pred     = model(x_cat, x_num)
            val_loss += criterion(pred, y).item()
    val_loss /= len(test_loader)

    scheduler.step(val_loss)

    # 최적 모델 저장
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_recommender.pth')

    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1:3d}/{NUM_EPOCHS} | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

print(f"\n✅ 학습 완료 | Best Val Loss: {best_val_loss:.4f}")

# ─────────────────────────────────────────────
# 7. 전처리 객체 저장 (배포용)
# ─────────────────────────────────────────────
with open('preprocessors.pkl', 'wb') as f:
    pickle.dump({
        'label_encoders': label_encoders,
        'scaler':         scaler,
        'target_scaler':  target_scaler,
        'cat_features':   cat_features,
        'num_features':   num_features,
    }, f)

print("💾 모델 및 전처리 객체 저장 완료")
