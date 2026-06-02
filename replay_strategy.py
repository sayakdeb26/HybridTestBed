import numpy as np
from typing import Tuple

class ReplayBuffer:
    def __init__(self, max_size: int = 300, balanced: bool = True, weak_class_ids: Tuple[int, ...] = (3, 4, 5)):
        self.max_size = max_size
        self.balanced = balanced
        self.weak_class_ids = weak_class_ids
        
        self.X_buffer = None
        self.y_buffer = None
        
    def size(self) -> int:
        if self.X_buffer is None:
            return 0
        return len(self.X_buffer)
        
    def add_batch(self, X: np.ndarray, y: np.ndarray):
        if self.X_buffer is None:
            self.X_buffer = X.copy()
            self.y_buffer = y.copy()
        else:
            self.X_buffer = np.concatenate([self.X_buffer, X], axis=0)
            self.y_buffer = np.concatenate([self.y_buffer, y], axis=0)
            
        # Manage size and balance
        if len(self.X_buffer) > self.max_size:
            if self.balanced:
                self._balance_buffer()
            else:
                # FIFO replacement
                self.X_buffer = self.X_buffer[-self.max_size:]
                self.y_buffer = self.y_buffer[-self.max_size:]
                
    def _balance_buffer(self):
        unique_classes = np.unique(self.y_buffer)
        n_classes = len(unique_classes)
        
        # Calculate samples per class
        samples_per_class = self.max_size // n_classes
        
        X_balanced = []
        y_balanced = []
        
        for c in unique_classes:
            idx = np.where(self.y_buffer == c)[0]
            if len(idx) == 0:
                continue
                
            # If it's a weak class, maybe we want to keep more
            n_keep = min(len(idx), samples_per_class)
            if c in self.weak_class_ids:
                n_keep = min(len(idx), int(samples_per_class * 1.5))
                
            keep_idx = np.random.choice(idx, n_keep, replace=False)
            X_balanced.append(self.X_buffer[keep_idx])
            y_balanced.append(self.y_buffer[keep_idx])
            
        self.X_buffer = np.concatenate(X_balanced, axis=0)
        self.y_buffer = np.concatenate(y_balanced, axis=0)
        
        # If still over limit, truncate randomly
        if len(self.X_buffer) > self.max_size:
            idx = np.random.choice(len(self.X_buffer), self.max_size, replace=False)
            self.X_buffer = self.X_buffer[idx]
            self.y_buffer = self.y_buffer[idx]
            
    def sample(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        if self.X_buffer is None or len(self.X_buffer) == 0:
            return np.array([]), np.array([])
            
        n_sample = min(n, len(self.X_buffer))
        idx = np.random.choice(len(self.X_buffer), n_sample, replace=False)
        return self.X_buffer[idx], self.y_buffer[idx]
