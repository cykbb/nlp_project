from tkinter import E
from typing import Any, Callable, List, Optional
from d2l.base.evaluator import Evaluator
from tqdm.auto import tqdm # type: ignore
import torch

class RNNEvaluator(Evaluator):
    def loss (self, test_data_loader: Any) -> float:
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        with torch.no_grad():
            for X, y in test_data_loader:
                y_hat, _ = self.model.forward(X)
                batch_loss = self.model.loss(y_hat, y)
                batch_size = y.shape[0]
                total_loss += batch_loss.item() * batch_size
                total_samples += batch_size
        if total_samples == 0:
            raise ValueError("Test dataloader produced zero samples.")
        return total_loss / total_samples