import torch
import torch.nn as nn
import numpy as np
from typing import Dict
from .base_strategy import BaseStrategy, StrategyConfig

class EWCStrategy(BaseStrategy):
    def __init__(self, config: StrategyConfig, ewc_lambda: float = 0.3, fisher_samples: int = 150, online_alpha: float = 0.5):
        super().__init__(config)
        self.ewc_lambda = ewc_lambda
        self.fisher_samples = fisher_samples
        self.online_alpha = online_alpha
        
        # EWC state
        self.fisher = {}
        self.optpar = {}
        
    def train_on_batch(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        self.model.train()
        
        inputs = torch.tensor(X, dtype=torch.float32).to(self.device)
        targets = torch.tensor(y, dtype=torch.long).to(self.device)
        
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)
        
        # Add EWC penalty
        ewc_loss = torch.tensor(0.0).to(self.device)
        if self.fisher:
            for name, param in self.model.named_parameters():
                if name in self.fisher:
                    # L_ewc = lambda/2 * F * (theta - theta_star)^2
                    ewc_loss += (self.fisher[name] * (param - self.optpar[name]) ** 2).sum()
            loss = loss + (self.ewc_lambda / 2.0) * ewc_loss
            
        loss.backward()
        self.optimizer.step()
        
        # Calculate accuracy
        _, preds = torch.max(outputs, 1)
        acc = (preds == targets).float().mean().item()
        
        return {
            "loss": loss.item(),
            "accuracy": acc,
            "ewc_loss": ewc_loss.item()
        }
        
    def on_task_end(self, X: np.ndarray, y: np.ndarray):
        # Compute Fisher Information Matrix
        self.model.eval()
        
        # Prepare parameters
        new_fisher = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_fisher[name] = torch.zeros_like(param.data)
                
        # Sample data to compute Fisher
        n_samples = min(len(X), self.fisher_samples)
        indices = np.random.choice(len(X), n_samples, replace=False)
        X_sampled = X[indices]
        y_sampled = y[indices]
        
        inputs = torch.tensor(X_sampled, dtype=torch.float32).to(self.device)
        targets = torch.tensor(y_sampled, dtype=torch.long).to(self.device)
        
        for i in range(n_samples):
            self.model.zero_grad()
            out = self.model(inputs[i:i+1])
            
            # Use empirical Fisher
            loss = self.criterion(out, targets[i:i+1])
            loss.backward()
            
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    new_fisher[name] += (param.grad.data ** 2) / n_samples
                    
        # Update running Fisher and save current parameters as optpar
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if name in self.fisher:
                    # Online EWC update rule
                    self.fisher[name] = self.online_alpha * self.fisher[name] + (1 - self.online_alpha) * new_fisher[name]
                else:
                    self.fisher[name] = new_fisher[name]
                self.optpar[name] = param.data.clone()
