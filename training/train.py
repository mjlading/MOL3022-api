import os, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from model.model import CNNModel, OUTPUT_SIZE

# Setup
scaler = GradScaler()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on", device)

# Hyperparameters and mappings
SW, HW = 17, 17 // 2
residues = ['I','E','N','F','S','D','G','W','T','L','Q','C','B','M','V','K','X','H','P','Z','A','Y','R']
structures = ['_', 'H', '?', 'E', 'G', 'S']
res_to_idx = {r: i for i, r in enumerate(residues)}
struc_to_idx = {s: i for i, s in enumerate(structures)}

# Data loading and processing
X, y = [], []
data_path = "data/513_distribute"
for fname in os.listdir(data_path):
    with open(os.path.join(data_path, fname)) as f:
        lines = f.readlines()
    r_seq = [r for r in lines[0].split(":")[1].strip().split(",") if r]
    r_seq = ["X"] * HW + r_seq + ["X"] * HW
    r_enc = [res_to_idx[r] for r in r_seq]
    s_enc = [struc_to_idx[s] for s in lines[5].split(":")[1].strip().split(",")[:-1]]
    for i in range(HW, len(r_enc) - HW):
        X.append(r_enc[i - HW:i + HW + 1])
        y.append(s_enc[i - HW])
print("Total samples:", len(X))

# Split and convert to tensors
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.1, random_state=42)
to_tensor = lambda data: torch.tensor(np.stack(data), dtype=torch.long)
X_train, X_val, X_test = map(to_tensor, (X_train, X_val, X_test))
y_train = torch.tensor(y_train, dtype=torch.long)
y_val   = torch.tensor(y_val, dtype=torch.long)
y_test  = torch.tensor(y_test, dtype=torch.long)

batch_size = 2048
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size, shuffle=True)
val_loader   = DataLoader(TensorDataset(X_val, y_val), batch_size)
test_loader  = DataLoader(TensorDataset(X_test, y_test), batch_size)

model = CNNModel(num_residues=len(residues), embedding_dim=50).to(device)
optimizer = Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = CosineAnnealingLR(optimizer, T_max=200)
criterion = nn.CrossEntropyLoss()

# Track training
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

# Training loop with early stopping
best_loss, patience, ctr = float('inf'), 20, 0
for epoch in range(200):
    model.train()
    loss_sum, correct, total = 0, 0, 0
    for data, targets in train_loader:
        data, targets = data.to(device), targets.to(device)
        optimizer.zero_grad()
        with autocast():
            scores = model(data)
            loss = criterion(scores, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        loss_sum += loss.item()
        preds = scores.max(1)[1]
        correct += (preds == targets).sum().item()
        total += targets.size(0)
    train_loss, train_acc = loss_sum / len(train_loader), correct / total * 100

    model.eval()
    loss_val, correct_val, total_val = 0, 0, 0
    with torch.no_grad():
        for data, targets in val_loader:
            data, targets = data.to(device), targets.to(device)
            scores = model(data)
            loss_val += criterion(scores, targets).item()
            preds = scores.max(1)[1]
            correct_val += (preds == targets).sum().item()
            total_val += targets.size(0)
    val_loss, val_acc = loss_val / len(val_loader), correct_val / total_val * 100

    # Track progress
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accuracies.append(train_acc)
    val_accuracies.append(val_acc)

    print(f"Epoch {epoch+1}/200 - Train: {train_loss:.4f}, {train_acc:.2f}% | Val: {val_loss:.4f}, {val_acc:.2f}%")

    if val_loss < best_loss:
        best_loss, ctr = val_loss, 0
        torch.save(model.state_dict(), "best_model.pt")
    elif (ctr := ctr + 1) >= patience:
        print("Early stopping.")
        break
    scheduler.step()

# Plot training curves
plt.figure()
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training vs Validation Loss')
plt.legend()
plt.tight_layout()
plt.savefig("loss_curve.png")

plt.figure()
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(val_accuracies, label='Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Training vs Validation Accuracy')
plt.legend()
plt.tight_layout()
plt.savefig("accuracy_curve.png")

# Evaluation and ROC Curve
model.load_state_dict(torch.load("best_model.pt"))
model.eval()
num_correct, num_samples = 0, 0
all_probs, all_targets = [], []
with torch.no_grad():
    for data, targets in test_loader:
        data, targets = data.to(device), targets.to(device)
        scores = model(data)
        probs = F.softmax(scores, dim=1)
        all_probs.append(probs.cpu().numpy())
        all_targets.append(targets.cpu().numpy())
        num_correct += (scores.max(1)[1] == targets).sum().item()
        num_samples += targets.size(0)
print(f"Test Accuracy: {num_correct / num_samples * 100:.2f}%")

all_probs = np.concatenate(all_probs)
all_targets = np.concatenate(all_targets)
all_bin = label_binarize(all_targets, classes=range(OUTPUT_SIZE))
fpr, tpr, _ = roc_curve(all_bin.ravel(), all_probs.ravel())
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], '--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.tight_layout()
plt.savefig("roc_curve.png")
