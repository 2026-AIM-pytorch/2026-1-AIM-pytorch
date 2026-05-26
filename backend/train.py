import torch, numpy as np
import torch.nn as nn
from dataset import prepare_data, save_preprocessors, cat_features, num_features
from model import AdvancedRecommender

def train(epochs=50, batch_size=128, lr=1e-3, ckpt='best_recommender.pth', pp_path='preprocessors.pkl'):
    torch.manual_seed(42); np.random.seed(42)

    train_loader, test_loader, pp = prepare_data(batch_size=batch_size)
    label_encoders = pp['label_encoders']

    vocab_sizes = {col: len(label_encoders[col].classes_) for col in cat_features}
    model       = AdvancedRecommender(vocab_sizes, len(num_features))
    optimizer   = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler   = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = epochs, eta_min = 1e-5)
    criterion   = nn.HuberLoss(delta=1.0)

    best_val_loss = float('inf')

    for epoch in range(epochs):
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

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_cat, x_num, y in test_loader:
                pred = model(x_cat, x_num)
                val_loss += criterion(pred, y).item()
        val_loss /= len(test_loader)
        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), ckpt)
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
    
    save_preprocessors(pp, pp_path)
    print(f"\n✅ 학습 완료 | Best Val Loss: {best_val_loss:.4f}\n 저장 완료")

if __name__ == '__main__':
    train()