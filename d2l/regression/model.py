import torch
from torch import nn
from d2l.base.model import RegressionModel

class LinearRegression(RegressionModel):
    def __init__(self, 
                 num_features: int, 
                 rng: torch.Generator = torch.Generator().manual_seed(42)) -> None:
        super().__init__()
        self.num_features: int = num_features
        self.rng: torch.Generator = rng
        self.w: nn.Parameter = nn.Parameter(torch.normal(0, 0.01, (num_features, 1), generator=self.rng))
        self.b: nn.Parameter = nn.Parameter(torch.zeros(1))
        
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return X @ self.w + self.b

    def loss(self, y_hat: torch.Tensor, y) -> torch.Tensor:
        loss = (y_hat - y) ** 2 / 2
        return loss.mean()

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        return self.forward(X)
    
class LinearRegressionL2(LinearRegression):
    def __init__(self, 
                 num_features: int, 
                 weight_decay: float = 0.01,
                 rng: torch.Generator = torch.Generator().manual_seed(42)) -> None:
        super().__init__(num_features, rng)
        self.weight_decay: float = weight_decay

    def l2_penalty(self, weight: torch.Tensor) -> torch.Tensor:
        return 0.5 * weight.pow(2).sum()

    def loss(self, y_hat: torch.Tensor, y) -> torch.Tensor:
        l2_penalty = self.l2_penalty(self.w)
        loss = (y_hat - y) ** 2 / 2 + self.weight_decay * l2_penalty
        return loss.mean()

class LinearRegressionTorch(RegressionModel):
    def __init__(self, 
                 num_features: int,
                 rng: torch.Generator = torch.Generator().manual_seed(42)) -> None:
        super().__init__()
        self.num_features: int = num_features
        self.rng: torch.Generator = rng
        
        linear = nn.Linear(num_features, 1)
        torch.nn.init.normal_(linear.weight, 0, 0.01, generator=self.rng)
        torch.nn.init.zeros_(linear.bias)
        
        self.net = nn.Sequential(linear)

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        return self.forward(X)
    
    def loss(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:   
        return nn.functional.mse_loss(y_hat, y)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.net(X)
