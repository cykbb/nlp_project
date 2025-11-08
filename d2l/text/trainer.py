from typing import Any, Callable, List, Optional
from d2l.base.trainer import Trainer
from tqdm.auto import tqdm # type: ignore
import torch

class RNNTrainer(Trainer):
    def __init__(
        self,
        model: Any,
        optimizer: Any,
        on_train_epoch_end: Optional[Callable[[Any, int, List[float]], Any]] = None, 
        is_train_progress_leave: bool = True
    ) -> None:
        super().__init__(model, optimizer, on_train_epoch_end, is_train_progress_leave)
        
    def _train_single_epoch(self, train_data_loader: Any) -> Any:
        self.model.train()
        batch_losses: List[float] = []
        for X, y in tqdm(train_data_loader, desc="Batch", leave=self.is_train_progress_leave):
            y_hat, final_states = self.model.forward(X)
            loss = self.model.loss(y_hat, y)
            loss.backward()
            batch_losses.append(loss.item())
            # clip gradients to prevent gradient explosion
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.optimizer.zero_grad()
        return batch_losses