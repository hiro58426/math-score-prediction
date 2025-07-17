# Math Score Prediction

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt

# データ読み込み
df = pd.read_csv("StudentsPerformance.csv")

# カテゴリ変数の数値化
label_cols = ["gender", "race/ethnicity", "parental level of education", "lunch", "test preparation course"]
for col in label_cols:
    df[col] = LabelEncoder().fit_transform(df[col])

# 特徴量と目的変数
X = df.drop(columns=["reading score", "writing score", "math score"])
y = df["math score"]

# 特徴量を標準化、目的変数を0〜1に正規化
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

y_min, y_max = y.min(), y.max()
y_scaled = (y - y_min) / (y_max - y_min)

# テンソル変換
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y_scaled.values.reshape(-1, 1), dtype=torch.float32)

# 学習・テスト分割
X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

# ニューラルネットワーク定義（ReLU使用）
class FourLayerNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.l1 = torch.nn.Linear(input_size, hidden_size)
        self.l2 = torch.nn.Linear(hidden_size, hidden_size)
        self.l3 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        return self.l3(x)

model = FourLayerNN(X_train.shape[1], 30, 1)

# 学習関数（学習率低め）
def train_model(model, input, target):
    dataset = torch.utils.data.TensorDataset(input, target)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(3000):
        for xb, yb in loader:
            pred = model(xb)
            loss = torch.nn.functional.mse_loss(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch % 500 == 0:
            with torch.inference_mode():
                total_loss = torch.nn.functional.mse_loss(model(input), target)
                print(f"Epoch {epoch}: Loss = {total_loss.item():.4f}")

# モデル学習
train_model(model, X_train, y_train)

# 評価（正規化を元に戻して評価）
with torch.inference_mode():
    pred_scaled = model(X_test)
    pred = pred_scaled * (y_max - y_min) + y_min
    true = y_test * (y_max - y_min) + y_min
    mse = torch.nn.functional.mse_loss(pred, true)
    rmse = torch.sqrt(mse)
    print(f"\n✅ Test RMSE: {rmse.item():.2f}")

# グラフ描画
plt.figure(figsize=(6, 6))
plt.scatter(true, pred, alpha=0.6)
plt.plot([0, 100], [0, 100], 'r--')
plt.xlabel("Actual Math Score")
plt.ylabel("Predicted Math Score")
plt.title("Actual vs Predicted Math Scores (Improved)")
plt.grid(True)
plt.tight_layout()
plt.show()

# 進捗中の変更
