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
