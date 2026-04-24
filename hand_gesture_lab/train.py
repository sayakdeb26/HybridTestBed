import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import numpy as np
import os
import json
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns


# =============================================================================
# Dataset with proper feature engineering
# =============================================================================
class GestureDataset(Dataset):
    def __init__(self, x_path, y_path, augment=False):
        print(f"Loading data from {x_path}...")
        X_raw = np.load(x_path).astype(np.float32)
        y_raw = np.load(y_path)
        
        # 1. Labels
        if len(y_raw.shape) > 1 and y_raw.shape[1] > 1:
            self.y = np.argmax(y_raw, axis=1)
        else:
            self.y = y_raw
        self.y = self.y.astype(np.int64)
        
        # --- 5-Class Filtering & Re-mapping ---
        target_classes = {
            16: 0,  # Swiping Left
            17: 1,  # Swiping Right
            18: 2,  # Swiping Up
            15: 3,  # Swiping Down
            5: 4    # Pushing Hand Away
        }
        
        mask = np.isin(self.y, list(target_classes.keys()))
        self.y = self.y[mask]
        X_raw = X_raw[mask]
        
        self.y = np.vectorize(target_classes.get)(self.y)
        
        unique, counts = np.unique(self.y, return_counts=True)
        print("Class distribution after filtering:")
        for u, c in zip(unique, counts):
            print(f"  Class {u}: {c} samples")
        # --------------------------------------
        
        # 2. Separate pose and hand features for independent processing
        base_pos = X_raw[:, :, :60]   # Position features
        
        # 3. Compute velocity
        velocity = np.zeros_like(base_pos)
        velocity[:, 1:, :] = base_pos[:, 1:, :] - base_pos[:, :-1, :]
        
        # 4. Compute acceleration
        acceleration = np.zeros_like(base_pos)
        acceleration[:, 2:, :] = velocity[:, 2:, :] - velocity[:, 1:-1, :]
        
        # Clamp velocity and acceleration
        velocity = np.clip(velocity, -3.0, 3.0)
        acceleration = np.clip(acceleration, -3.0, 3.0)
        
        # --- NEW ADVANCED FEATURES ---
        # A. Motion Magnitude (Overall intensity)
        motion_mag = np.linalg.norm(velocity, axis=-1, keepdims=True)
        
        # B. Direction Encoding for Wrists (Normalized vectors)
        # Left Wrist: dims 12:15, Right Wrist: dims 15:18
        l_wrist_vel = velocity[:, :, 12:15]
        r_wrist_vel = velocity[:, :, 15:18]
        
        l_wrist_dir = l_wrist_vel / (np.linalg.norm(l_wrist_vel, axis=-1, keepdims=True) + 1e-6)
        r_wrist_dir = r_wrist_vel / (np.linalg.norm(r_wrist_vel, axis=-1, keepdims=True) + 1e-6)
        
        # C. Push Detection Feature (Depth proxy via wrist-to-shoulder distance)
        # Since origin is shoulder midpoint, wrist norm IS the wrist-to-shoulder distance
        l_wrist_dist = np.linalg.norm(base_pos[:, :, 12:15], axis=-1, keepdims=True)
        r_wrist_dist = np.linalg.norm(base_pos[:, :, 15:18], axis=-1, keepdims=True)
        
        delta_l_depth = np.zeros_like(l_wrist_dist)
        delta_l_depth[:, 1:, :] = l_wrist_dist[:, 1:, :] - l_wrist_dist[:, :-1, :]
        
        delta_r_depth = np.zeros_like(r_wrist_dist)
        delta_r_depth[:, 1:, :] = r_wrist_dist[:, 1:, :] - r_wrist_dist[:, :-1, :]
        
        # --- NEW BODY-RELATIVE ARM EXTENSION ---
        # D. Arm Extension (Scale-Invariant)
        # Left shoulder: dims 0:3, Right shoulder: dims 3:6
        shoulder_width = np.linalg.norm(base_pos[:, :, 0:3] - base_pos[:, :, 3:6], axis=-1, keepdims=True)
        
        # Average the left and right arm extension
        arm_extension = (l_wrist_dist + r_wrist_dist) / 2.0
        
        # Normalize by shoulder width
        arm_extension = arm_extension / (shoulder_width + 1e-6)
        
        # Clamp to avoid spikes from missing shoulder joints
        arm_extension = np.clip(arm_extension, 0.0, 5.0)
        
        # Temporal change of arm extension
        delta_arm = np.zeros_like(arm_extension)
        delta_arm[:, 1:, :] = arm_extension[:, 1:, :] - arm_extension[:, :-1, :]
        delta_arm = np.clip(delta_arm, -1.0, 1.0)
        
        # Concatenate everything
        self.X = np.concatenate([
            base_pos, 
            velocity, 
            acceleration,
            motion_mag,         # 1 dim
            l_wrist_dir,        # 3 dims
            r_wrist_dir,        # 3 dims
            delta_l_depth,      # 1 dim
            delta_r_depth,      # 1 dim
            arm_extension,      # 1 dim
            delta_arm           # 1 dim
        ], axis=2)
        
        # Sequence-Level Standardization
        seq_mean = np.mean(self.X, axis=1, keepdims=True)
        seq_std = np.std(self.X, axis=1, keepdims=True)
        seq_std[seq_std == 0] = 1e-6
        self.X = (self.X - seq_mean) / seq_std
        
        # Final clamp
        self.X = np.clip(self.X, -5.0, 5.0).astype(np.float32)
        
        self.augment = augment
        print(f"Dataset loaded. X shape: {self.X.shape}, y shape: {self.y.shape}")
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx])
        y = torch.tensor(self.y[idx])
        
        if self.augment:
            if torch.rand(1).item() < 0.5:
                x = x + torch.randn_like(x) * 0.02
            if torch.rand(1).item() < 0.3:
                T = x.shape[0]
                warp = torch.linspace(0, T-1, T) + torch.randn(T) * 0.5
                warp = warp.clamp(0, T-1).long()
                x = x[warp]
            if torch.rand(1).item() < 0.2:
                feat_dim = x.shape[1]
                start = torch.randint(0, max(1, feat_dim - 18), (1,)).item()
                x[:, start:start+18] = 0
            if torch.rand(1).item() < 0.3:
                x = x.flip(0)
                
        return x, y

class AugmentedSubset(Dataset):
    def __init__(self, subset, augment=False):
        self.subset = subset
        self.augment = augment
        
    def __len__(self):
        return len(self.subset)
        
    def __getitem__(self, idx):
        x, y = self.subset[idx]
        if self.augment:
            # Gaussian noise (always apply when augmenting)
            x = x + torch.randn_like(x) * 0.01
            if torch.rand(1).item() < 0.3:
                T = x.shape[0]
                warp = torch.linspace(0, T-1, T) + torch.randn(T) * 0.5
                warp = warp.clamp(0, T-1).long()
                x = x[warp]
            if torch.rand(1).item() < 0.2:
                feat_dim = x.shape[1]
                start = torch.randint(0, max(1, feat_dim - 18), (1,)).item()
                x[:, start:start+18] = 0
            if torch.rand(1).item() < 0.3:
                x = x.flip(0)
        return x, y

class GestureLSTM(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(GestureLSTM, self).__init__()
        
        # Reduced model size (128->96, 64->48) + Dropout 0.4
        self.lstm1 = nn.LSTM(input_dim, 96, batch_first=True, bidirectional=True)
        self.ln1 = nn.LayerNorm(192)
        self.dropout1 = nn.Dropout(0.4)
        
        self.lstm2 = nn.LSTM(192, 48, batch_first=True, bidirectional=True)
        self.ln2 = nn.LayerNorm(96)
        self.dropout2 = nn.Dropout(0.4)
        
        # Classification head (applied per frame)
        self.classifier = nn.Sequential(
            nn.Linear(96, 48),
            nn.BatchNorm1d(48),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(48, num_classes)
        )
        
    def forward(self, x):
        B, T, _ = x.size()
        
        # Layer 1
        out, _ = self.lstm1(x)
        out = self.ln1(out)
        out = self.dropout1(out)
        
        # Layer 2
        out, _ = self.lstm2(out)
        out = self.ln2(out)
        out = self.dropout2(out)
        
        # Flatten for frame-level classification
        out_flat = out.contiguous().view(B * T, -1)
        logits_flat = self.classifier(out_flat)
        logits = logits_flat.view(B, T, -1)
        
        # Temporal logit smoothing
        smoothed = torch.zeros_like(logits)
        smoothed[:, 0, :] = logits[:, 0, :]
        alpha = 0.6
        
        for t in range(1, T):
            smoothed[:, t, :] = alpha * logits[:, t, :] + (1 - alpha) * smoothed[:, t-1, :]
            
        # Final class decision is based on the smoothed logit at the last frame
        return smoothed[:, -1, :]


# =============================================================================
# Training
# =============================================================================
def train_model():
    data_dir = '/home/sayak/HybridTestBed/hand_gesture_lab/data/processed/train'
    x_path = os.path.join(data_dir, 'X_fixed.npy')
    y_path = os.path.join(data_dir, 'y.npy')
    
    batch_size = 64
    epochs = 150
    learning_rate = 0.0005
    val_split = 0.2
    patience = 20
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Disable cuDNN to fix version mismatch (9.19.0 vs 9.21.1)
    torch.backends.cudnn.enabled = False
    
    full_dataset = GestureDataset(x_path, y_path, augment=False)
    input_dim = full_dataset.X.shape[2]
    classes = np.unique(full_dataset.y)
    num_classes = len(classes)
    
    # Stratified-ish split
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)
    
    # Wrap with augmentation for training only
    train_augmented = AugmentedSubset(train_dataset, augment=True)
    
    train_loader = DataLoader(train_augmented, batch_size=batch_size, shuffle=True, 
                              pin_memory=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                            pin_memory=True, num_workers=2)
    
    # Class weights
    y_train = full_dataset.y[train_dataset.indices]
    class_weights_np = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weights = torch.tensor(class_weights_np, dtype=torch.float32).to(device)
    
    model = GestureLSTM(input_dim, num_classes).to(device)
    
    # Label smoothing + class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    
    # AdamW with weight decay for better regularization
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # Cosine annealing for smoother LR schedule
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)
    
    best_val_acc = 0.0
    epochs_no_improve = 0
    os.makedirs('weights', exist_ok=True)
    
    best_all_preds = []
    best_all_targets = []
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Training samples: {train_size}, Validation samples: {val_size}")
    print(f"Input dim: {input_dim}, Num classes: {num_classes}\n")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            
            # Gradient clipping for training stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
            
            _, preds = torch.max(outputs, 1)
            train_correct += (preds == batch_y).sum().item()
            train_total += batch_y.size(0)
            
        scheduler.step()
        train_loss /= len(train_loader)
        train_acc = train_correct / train_total
        
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []
        all_probs = []
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                
                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(batch_y.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                
        val_loss /= len(val_loader)
        val_acc = accuracy_score(all_targets, all_preds)
        
        # Top-2 accuracy
        all_probs_np = np.array(all_probs)
        all_targets_np = np.array(all_targets)
        top2_preds = np.argsort(all_probs_np, axis=1)[:, -2:]
        top2_correct = np.array([t in p for t, p in zip(all_targets_np, top2_preds)])
        top2_acc = np.mean(top2_correct)
        
        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Top-2: {top2_acc:.4f} | LR: {lr:.6f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_top2 = top2_acc
            best_all_preds = all_preds
            best_all_targets = all_targets
            epochs_no_improve = 0
            torch.save(model.state_dict(), 'weights/best_lstm_model.pth')
            print("  --> Best model saved!")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs!")
                break
                
    print(f"\nTraining complete. Best Validation Accuracy: {best_val_acc:.4f} | Best Top-2: {best_top2:.4f}")
    
    # Generate Confusion Matrix
    label_names = ['Swipe Left', 'Swipe Right', 'Swipe Up', 'Swipe Down', 'Push Away']
    cm = confusion_matrix(best_all_targets, best_all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=label_names, yticklabels=label_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title(f'Confusion Matrix (Val Acc: {best_val_acc:.4f})')
    plt.savefig('confusion_matrix.png', bbox_inches='tight', dpi=150)
    print("Saved confusion matrix to confusion_matrix.png")
    
    # Per-class accuracy
    print("\nClassification Report:")
    print(classification_report(best_all_targets, best_all_preds, target_names=label_names))

if __name__ == '__main__':
    train_model()
