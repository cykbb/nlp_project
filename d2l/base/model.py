import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple, cast


class Model(nn.Module, ABC):
    def __init__(self) -> None:
        super().__init__()

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load(self, path: str, map_location: Optional[torch.device] = None) -> None:
        state_dict = torch.load(path, map_location=map_location)
        self.load_state_dict(state_dict)

    @abstractmethod
    def loss(self, y_hat: Any, y: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError


class ClassifierModel(Model, ABC):
    def predict(self, X: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        with torch.no_grad():
            output = self.forward(X, *args, **kwargs)
        return cast(torch.Tensor, output).argmax(dim=1)

class RegressionModel(Model, ABC):
    def predict(self, X: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        return cast(torch.Tensor, self.forward(X, *args, **kwargs))

class LanguageModel(Model, ABC):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int) -> None:
        super().__init__()
        self.vocab_size = input_size
        self.num_hiddens = hidden_size
        self.num_layers = num_layers

    def init_state(self, batch_size: int, device: torch.device) -> Any:
        return torch.zeros((self.num_layers, batch_size, self.num_hiddens), device=device)

    def one_hot(self, X: torch.Tensor) -> torch.Tensor:
        return F.one_hot(X.T, num_classes=self.vocab_size).float()

    def loss(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        logits_reshaped = y_hat.reshape(-1, self.vocab_size)
        target_reshaped = y.reshape(-1)
        return F.cross_entropy(logits_reshaped, target_reshaped)

    def predict(
        self,
        X: torch.Tensor,
        states: Optional[Any] = None,
    ) -> Tuple[torch.Tensor, Any]:
        self.eval()
        with torch.no_grad():
            output_logits, final_states = self.forward(X, states)
            predicted_indices = output_logits.argmax(dim=-1)
        return predicted_indices, final_states
