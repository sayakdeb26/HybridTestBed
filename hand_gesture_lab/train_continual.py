import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Add parent path to import HybridTestBed packages
sys.path.append('/home/sayak')
sys.path.append('/home/sayak/HybridTestBed/hand_gesture_lab')

from HybridTestBed.mixed_strategy import MixedStrategy, StrategyConfig
from train import GestureLSTM

def train_continual(task_name="DS1", model_path="/home/sayak/HybridTestBed/hand_gesture_lab/weights/best_lstm_model.pth", out_model_path="/home/sayak/HybridTestBed/hand_gesture_lab/weights/best_lstm_model_continual.pth"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Disable cuDNN to prevent version mismatch crash
    torch.backends.cudnn.enabled = False
    
    # 1. Load LSTM Model
    # input_dim = 296 (30 frames x 296 dimensions)
    model = GestureLSTM(input_dim=296, num_classes=6).to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded base model weights from {model_path}")
    else:
        print(f"Warning: Base model weights not found at {model_path}")
        
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # 2. Setup continual learning strategy config and MixedStrategy
    config = StrategyConfig(model, optimizer, criterion, device, augment_replay=True)
    strategy = MixedStrategy(config)
    
    # Load EWC + Replay Buffer state if it exists
    cl_state_path = "/home/sayak/HybridTestBed/hand_gesture_lab/weights/cl_state.pth"
    if strategy.load_state(cl_state_path):
        print(f"Successfully loaded continual learning state (Fisher + Replay Buffer) from {cl_state_path}")
    else:
        print(f"No existing continual learning state found at {cl_state_path}. Starting fresh.")
    
    # 3. Simulate/Load task training data
    print(f"Starting continual learning retraining loop for task: {task_name}")
    
    # Generate dummy training data matching LSTM shapes for testing strategy functionality
    X_task = np.random.randn(100, 30, 296).astype(np.float32)
    y_task = np.random.randint(0, 6, size=(100,)).astype(np.int64)
    
    # 4. Retraining Loop
    batch_size = 32
    n_batches = len(X_task) // batch_size
    
    for epoch in range(5):
        epoch_loss = 0.0
        epoch_acc = 0.0
        for b in range(n_batches):
            start = b * batch_size
            end = start + batch_size
            X_batch = X_task[start:end]
            y_batch = y_task[start:end]
            
            metrics = strategy.train_on_batch(X_batch, y_batch)
            epoch_loss += metrics["loss"]
            epoch_acc += metrics["accuracy"]
            
        print(f"Epoch {epoch+1}/5 | Loss: {epoch_loss/n_batches:.4f} | Acc: {epoch_acc/n_batches:.4f} | EWC Loss: {metrics.get('ewc_loss', 0.0):.4f} | Buffer: {metrics['buffer_size']}")
        
    # 5. Transition to next task (updates Fisher consolidation and saves replay buffer)
    strategy.on_task_end(X_task, y_task)
    
    # Save continual learning state
    strategy.save_state(cl_state_path)
    print(f"Successfully saved continual learning state (Fisher + Replay Buffer) to {cl_state_path}")
    
    # 6. Save weights
    os.makedirs(os.path.dirname(out_model_path), exist_ok=True)
    torch.save(model.state_dict(), out_model_path)
    print(f"Successfully saved consolidated continual learning weights to {out_model_path}")

if __name__ == '__main__':
    train_continual()
