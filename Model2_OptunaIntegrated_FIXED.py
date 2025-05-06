# Wide & Deep Model with Train/Val/Test Split, SMOTENC, Focal Loss, and Regularization

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_recall_curve
from imblearn.over_sampling import SMOTENC
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import joblib

# === Load Data ===
file_path = r"C:\Users\dylan\Desktop\Data Science Final\diabetes_prediction_dataset.csv"
df = pd.read_csv(file_path)

# === Feature Engineering ===
num_cols = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
cat_cols = ['gender', 'smoking_history']
target_col = 'diabetes'

df['age_bin'] = pd.cut(df['age'], bins=[0, 20, 40, 60, 80, 100], labels=False)
df['bmi_bin'] = pd.cut(df['bmi'], bins=[0, 18.5, 25, 30, 35, 100], labels=False)
df[target_col] = df[target_col].astype(np.int64)

# Encode categorical features
cat_embed_mappings = {}
for col in cat_cols:
    df[col] = df[col].astype(str)
    df[col + '_idx'], cat_embed_mappings[col] = pd.factorize(df[col])

df['age_smoke_cross'] = df['age_bin'].astype(str) + "_" + df['smoking_history']
df['age_smoke_cross'], _ = pd.factorize(df['age_smoke_cross'])

# Normalize numerical features
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

# Finalize features
wide_features = ['hypertension', 'heart_disease', 'age_bin', 'bmi_bin', 'age_smoke_cross'] + [col + '_idx' for col in cat_cols]
deep_cont = num_cols
deep_cat = [col + '_idx' for col in cat_cols]

X_wide = df[wide_features].astype(int)
X_cont = df[deep_cont]
X_cat = df[deep_cat].astype(int)
y = df[target_col].values

# === Split into Train / Validation / Test ===
X_all = pd.concat([X_wide, X_cont, X_cat], axis=1)
X_train_val, X_test, y_train_val, y_test = train_test_split(X_all, y, test_size=0.2, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, stratify=y_train_val, random_state=42)

# === Apply SMOTENC to training set ===
cat_idxs = list(range(len(wide_features))) + list(range(len(wide_features) + len(deep_cont), X_all.shape[1]))
smote = SMOTENC(categorical_features=cat_idxs, random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# === Partition features ===
w_end = len(wide_features)
c_end = w_end + len(deep_cont)

X_wide_train = X_train_res.iloc[:, :w_end]
X_cont_train = X_train_res.iloc[:, w_end:c_end]
X_cat_train = X_train_res.iloc[:, c_end:]

X_wide_val = X_val.iloc[:, :w_end]
X_cont_val = X_val.iloc[:, w_end:c_end]
X_cat_val = X_val.iloc[:, c_end:]

X_wide_test = X_test.iloc[:, :w_end]
X_cont_test = X_test.iloc[:, w_end:c_end]
X_cat_test = X_test.iloc[:, c_end:]

# === Save scalers/mappings ===
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(cat_embed_mappings, 'cat_mappings.pkl')

# === Dataset ===
class WideDeepDataset(Dataset):
    def __init__(self, Xw, Xc, Xcat, y):
        self.Xw = torch.tensor(Xw.values, dtype=torch.long)
        self.Xc = torch.tensor(Xc.values, dtype=torch.float32)
        self.Xcat = torch.tensor(Xcat.values, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self): return len(self.y)
    def __getitem__(self, idx):
        return {'wide': self.Xw[idx], 'deep_cont': self.Xc[idx], 'deep_cat': self.Xcat[idx], 'target': self.y[idx]}

train_ds = WideDeepDataset(X_wide_train, X_cont_train, X_cat_train, y_train_res)
val_ds = WideDeepDataset(X_wide_val, X_cont_val, X_cat_val, y_val)
test_ds = WideDeepDataset(X_wide_test, X_cont_test, X_cat_test, y_test)

train_loader = DataLoader(train_ds, batch_size=512, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=512)
test_loader = DataLoader(test_ds, batch_size=512)

# === Model ===
class WideDeepModel(nn.Module):
    def __init__(self, wide_in, cont_in, cat_cardinals, dropout=0.3776416914679301):
        super().__init__()
        self.wide = nn.Linear(wide_in, 1)
        embed_dims = [min(50, (card + 1) // 2) for card in cat_cardinals]
        self.embeds = nn.ModuleList([nn.Embedding(c, e) for c, e in zip(cat_cardinals, embed_dims)])
        self.bn = nn.BatchNorm1d(cont_in + sum(embed_dims))
        self.deep = nn.Sequential(
            nn.Linear(cont_in + sum(embed_dims), 166),
            nn.GELU(),
            nn.LayerNorm(166),  # Match 166 units
            nn.Dropout(dropout),
            nn.Linear(166, 74),
            nn.GELU(),
            nn.LayerNorm(74),  # Match 74 units
            nn.Dropout(dropout),
            nn.Linear(74, 52),
            nn.GELU(),
            nn.LayerNorm(52),  # Match 52 units
            nn.Dropout(dropout),
        )
        self.out = nn.Linear(52, 1)

    def forward(self, w, c, cat):
        w_out = self.wide(w.float())
        cat_embed = torch.cat([emb(cat[:, i]) for i, emb in enumerate(self.embeds)], dim=1)
        deep_input = self.bn(torch.cat([c, cat_embed], dim=1))
        d_out = self.deep(deep_input)
        return torch.sigmoid(self.out(d_out) + w_out).squeeze(1)

# === Focal Loss ===
class FocalLoss(nn.Module):
    def __init__(self, gamma=1.7580830878307798, alpha=0.3501675421289596):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
    def forward(self, inputs, targets):
        bce = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.where(targets == 1, inputs, 1 - inputs)
        return (self.alpha * (1 - pt) ** self.gamma * bce).mean()

# === Training ===
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = WideDeepModel(X_wide_train.shape[1], X_cont_train.shape[1], [df[col].nunique() for col in deep_cat]).to(DEVICE)
criterion = FocalLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0032567291741719888, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)

def evaluate(model, loader):
    model.eval()
    y_true, y_prob = [], []
    with torch.no_grad():
        for batch in loader:
            w, c, cat = batch['wide'].to(DEVICE), batch['deep_cont'].to(DEVICE), batch['deep_cat'].to(DEVICE)
            y = batch['target'].to(DEVICE)
            out = model(w, c, cat)
            y_prob.extend(out.cpu().numpy())
            y_true.extend(y.cpu().numpy())
    y_pred = (np.array(y_prob) > 0.5).astype(int)
    return roc_auc_score(y_true, y_prob), accuracy_score(y_true, y_pred), f1_score(y_true, y_pred), y_true, y_prob

best_auc, patience = 0, 0
for epoch in range(20):
    model.train()
    loss_sum = 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/20"):
        w, c, cat, y = batch['wide'].to(DEVICE), batch['deep_cont'].to(DEVICE), batch['deep_cat'].to(DEVICE), batch['target'].to(DEVICE)
        optimizer.zero_grad()
        out = model(w, c, cat)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()
    val_auc, val_acc, val_f1, _, _ = evaluate(model, val_loader)
    scheduler.step(1 - val_auc)
    print(f"Epoch {epoch+1}: Loss={loss_sum:.4f}, Val AUC={val_auc:.4f}, Acc={val_acc:.4f}, F1={val_f1:.4f}")
    if val_auc > best_auc:
        best_auc = val_auc
        patience = 0
        torch.save(model.state_dict(), 'best_wide_deep_model.pt')
        print("✅ Model saved!")
    else:
        patience += 1
        if patience >= 3:
            print("⏹️ Early stopping triggered.")
            break

# === Final Threshold Tuning & Test Evaluation ===
model.load_state_dict(torch.load('best_wide_deep_model.pt'))
_, _, _, y_val_true, y_val_prob = evaluate(model, val_loader)
precision, recall, thresholds = precision_recall_curve(y_val_true, y_val_prob)
f1s = 2 * precision * recall / (precision + recall + 1e-8)
best_thresh = thresholds[np.argmax(f1s)]
print(f"Optimal Threshold: {best_thresh:.2f}")

# === Evaluate on Test Set ===
_, _, _, y_test_true, y_test_prob = evaluate(model, test_loader)
y_test_pred = (np.array(y_test_prob) > best_thresh).astype(int)
print("\n=== Final Test Set Evaluation ===")
print(f"ROC AUC:  {roc_auc_score(y_test_true, y_test_prob):.4f}")
print(f"Accuracy: {accuracy_score(y_test_true, y_test_pred):.4f}")
print(f"F1 Score: {f1_score(y_test_true, y_test_pred):.4f}")
