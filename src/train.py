import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from project_utils import ProjectDataset
from model import OCRModel

# -----------------------------
# 1. LOAD DATASET
# -----------------------------
dataset = ProjectDataset(num_samples=5000)  
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# -----------------------------
# 2. CREATE MODEL
# -----------------------------
model = OCRModel()

# -----------------------------
# 3. LOSS + OPTIMIZER
# -----------------------------
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# -----------------------------
# 4. TRAINING LOOP
# -----------------------------
num_epochs = 10

for epoch in range(num_epochs):
    total_loss = 0
    all_preds = []
    all_labels = []

    for (image, type_vec), label in dataloader:
        
        label = label.long()

        # Forward
        outputs = model(image, type_vec)
        loss = criterion(outputs, label)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Collect metrics
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.tolist())
        all_labels.extend(label.tolist())

    # -----------------------------
    # METRICS
    # -----------------------------
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    print(f"\nEpoch {epoch+1}/{num_epochs}")
    print(f"Loss      = {total_loss:.4f}")
    print(f"Accuracy  = {acc:.4f}")
    print(f"Precision = {precision:.4f}")
    print(f"Recall    = {recall:.4f}")
    print(f"F1-score  = {f1:.4f}")

# -----------------------------
# 5. SAVE MODEL AUTOMATICALLY
# -----------------------------
save_path = os.path.join(os.path.dirname(__file__), "../saved_models/ocr_model.pth")
os.makedirs(os.path.dirname(save_path), exist_ok=True)
torch.save(model.state_dict(), save_path)
print(f"\nModel saved at {save_path}")
