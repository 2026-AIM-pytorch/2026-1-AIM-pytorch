import torch
import torch.nn as nn

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
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
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