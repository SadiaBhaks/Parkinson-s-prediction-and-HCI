import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

torch.manual_seed(42)
np.random.seed(42)

# ---------------------------------------------------------
# Load data (same task-level files as step7/step8)
# ---------------------------------------------------------
train = np.load("train_tasks.npz")
val = np.load("val_tasks.npz")
test = np.load("test_tasks.npz")

X_train_raw, y_train = train["X"], train["y"]
X_val_raw, y_val = val["X"], val["y"]
X_test_raw, y_test = test["X"], test["y"]

print("Raw shapes -> train:", X_train_raw.shape, "val:", X_val_raw.shape, "test:", X_test_raw.shape)

# ---------------------------------------------------------
# Normalize each channel using TRAINING statistics only
# (fit on train, apply to val/test -- never fit on test data)
# ---------------------------------------------------------
n_channels = X_train_raw.shape[2]
scaler = StandardScaler()
scaler.fit(X_train_raw.reshape(-1, n_channels))

def normalize(X):
    shape = X.shape
    X_flat = X.reshape(-1, n_channels)
    X_scaled = scaler.transform(X_flat)
    return X_scaled.reshape(shape)

X_train = normalize(X_train_raw)
X_val = normalize(X_val_raw)
X_test = normalize(X_test_raw)

# ---------------------------------------------------------
# PyTorch Dataset wrapper
# Conv1d expects (batch, channels, timesteps), so we transpose
# from (samples, timesteps, channels) -> (samples, channels, timesteps)
# ---------------------------------------------------------
class TaskDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32).permute(0, 2, 1)  # (N, C, T)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_ds = TaskDataset(X_train, y_train)
val_ds = TaskDataset(X_val, y_val)
test_ds = TaskDataset(X_test, y_test)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)

# ---------------------------------------------------------
# TCN building blocks
# ---------------------------------------------------------
class Chomp1d(nn.Module):
    """Removes extra padding added for causal convolution."""
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size] if self.chomp_size > 0 else x

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, dilation, dropout=0.2):
        super().__init__()
        padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2,
        )
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCN(nn.Module):
    def __init__(self, n_inputs, n_classes, channels=(32, 32, 64, 64), kernel_size=5, dropout=0.2):
        super().__init__()
        layers = []
        num_levels = len(channels)
        for i in range(num_levels):
            dilation = 2 ** i
            in_ch = n_inputs if i == 0 else channels[i - 1]
            out_ch = channels[i]
            layers.append(TemporalBlock(in_ch, out_ch, kernel_size, dilation, dropout))
        self.tcn = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(channels[-1], n_classes)

    def forward(self, x):
        out = self.tcn(x)
        out = self.global_pool(out).squeeze(-1)
        return self.fc(out)

# ---------------------------------------------------------
# Training setup
# ---------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

n_classes = len(np.unique(y_train))
model = TCN(n_inputs=n_channels, n_classes=n_classes).to(device)

# FIXED: gentler class-weight correction (square root softens it,
# instead of applying the full inverse-frequency correction, which
# overshot and caused the model to almost never predict PD)
class_counts = np.bincount(y_train)
raw_weights = len(y_train) / (n_classes * class_counts)
class_weights = torch.tensor(np.sqrt(raw_weights), dtype=torch.float32).to(device)
print("Class weights used:", class_weights.cpu().numpy())

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

# ---------------------------------------------------------
# Training loop with early stopping on validation balanced accuracy
# ---------------------------------------------------------
def evaluate(loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for Xb, yb in loader:
            Xb = Xb.to(device)
            logits = model(Xb)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(yb.numpy())
    acc = accuracy_score(all_labels, all_preds)
    bal_acc = balanced_accuracy_score(all_labels, all_preds)
    return acc, bal_acc, np.array(all_labels), np.array(all_preds)

# FIXED: more epochs and more patience, since training was still
# improving slowly when it stopped early last time
EPOCHS = 60
best_val_bal_acc = -1
best_state = None
patience = 12
patience_counter = 0

print("\nTraining TCN...")
for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0
    for Xb, yb in train_loader:
        Xb, yb = Xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(Xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * Xb.size(0)

    train_loss = total_loss / len(train_ds)
    val_acc, val_bal_acc, _, _ = evaluate(val_loader)

    print(f"Epoch {epoch:2d}/{EPOCHS}  train_loss={train_loss:.4f}  val_acc={val_acc:.3f}  val_balanced_acc={val_bal_acc:.3f}")

    if val_bal_acc > best_val_bal_acc:
        best_val_bal_acc = val_bal_acc
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch} (no val improvement for {patience} epochs)")
            break

model.load_state_dict(best_state)

# ---------------------------------------------------------
# Final evaluation on all three splits
# ---------------------------------------------------------
label_names = {0: "Healthy (HC)", 1: "Parkinson's (PD)", 2: "Other/DD"}

print("\n=========== FINAL TCN RESULTS ===========")
results = []
for name, loader in [("Train", train_loader), ("Validation", val_loader), ("Test (unseen)", test_loader)]:
    acc, bal_acc, labels, preds = evaluate(loader)
    print(f"\n--- {name} ---")
    print(f"Accuracy: {acc:.3f}   Balanced Accuracy: {bal_acc:.3f}")
    print(confusion_matrix(labels, preds, labels=[0, 1, 2]))
    print(classification_report(labels, preds, target_names=[label_names[k] for k in [0, 1, 2]], zero_division=0))
    results.append({"Model": "TCN", "Split": name, "Accuracy": acc, "Balanced_Accuracy": bal_acc})

results_df = pd.DataFrame(results)
results_df.to_csv("tcn_results.csv", index=False)

print("\n=========== SUMMARY ===========")
print(results_df[["Split", "Accuracy", "Balanced_Accuracy"]].round(3))
print("\nSaved to tcn_results.csv")

torch.save(model.state_dict(), "tcn_model.pt")
print("Saved trained model to tcn_model.pt")