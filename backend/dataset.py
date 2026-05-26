import os
import pandas as pd, torch, pickle
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'data')

cat_features = ['category', 'companion_type', 'place_name', 'mood',
                'district', 'crowdedness', 'mobility_preference', 'revisit']
num_features = ['age', 'google_rating', 'stay_time_minutes']
target       = 'schedule_satisfaction'

class TravelDataset(Dataset):
    def __init__(self, df, cat_cols, num_cols, target_col):
        self.X_cat = torch.tensor(df[cat_cols].values, dtype=torch.long)
        self.X_num = torch.tensor(df[num_cols].values, dtype=torch.float32)
        self.Y     = torch.tensor(df[target_col].values, dtype=torch.float32)

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        return self.X_cat[idx], self.X_num[idx], self.Y[idx]

def prepare_data(
    csv_path=os.path.join(DATA_DIR, 'travel_interactions.csv'),
    batch_size=64
):
    df = pd.read_csv(csv_path, encoding='utf-8-sig')

    for col in cat_features:
        df[col] = df[col].fillna('unknown').astype(str)

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df, test_df = train_df.copy(), test_df.copy()

    label_encoders = {}
    for col in cat_features:
        le = LabelEncoder().fit(df[col])
        train_df[col] = le.transform(train_df[col])
        test_df[col]  = le.transform(test_df[col])
        label_encoders[col] = le

    scaler = StandardScaler()
    train_df[num_features] = scaler.fit_transform(train_df[num_features])
    test_df[num_features]  = scaler.transform(test_df[num_features])

    target_scaler = StandardScaler()
    train_df[[target]] = target_scaler.fit_transform(train_df[[target]])
    test_df[[target]]  = target_scaler.transform(test_df[[target]])

    tr = DataLoader(TravelDataset(train_df, cat_features, num_features, target),
                    batch_size=batch_size, shuffle=True)
    te = DataLoader(TravelDataset(test_df,  cat_features, num_features, target),
                    batch_size=batch_size, shuffle=False)

    pp = {'label_encoders': label_encoders, 'scaler': scaler,
          'target_scaler': target_scaler,
          'cat_features': cat_features, 'num_features': num_features}

    return tr, te, pp

def save_preprocessors(pp, path='preprocessors.pkl'):
    with open(path, 'wb') as f: pickle.dump(pp, f)

def load_preprocessors(path='preprocessors.pkl'):
    with open(path, 'rb') as f: return pickle.load(f)