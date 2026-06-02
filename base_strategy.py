import torch
import numpy as np

class StrategyConfig:
    def __init__(self, model, optimizer, criterion, device, augment_replay=True):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.augment_replay = augment_replay

class BaseStrategy:
    def __init__(self, config: StrategyConfig):
        self.config = config
        self.model = config.model
        self.optimizer = config.optimizer
        self.criterion = config.criterion
        self.device = config.device
        
    def _augment_time_series(self, X: np.ndarray) -> np.ndarray:
        # Augment time series data
        # X shape: (B, T, F)
        # Apply Gaussian noise and random time warp
        B, T, F = X.shape
        augmented = X.copy()
        
        # Add small Gaussian noise
        noise = np.random.normal(0, 0.005, size=X.shape).astype(np.float32)
        augmented = augmented + noise
        
        # Random time shift/cutout
        for i in range(B):
            if np.random.rand() < 0.2:
                # Zero out a small segment of features
                start = np.random.randint(0, T - 5)
                augmented[i, start:start+5, :] = 0.0
                
        return augmented
