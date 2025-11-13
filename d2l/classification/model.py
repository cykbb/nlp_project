from typing import List, Tuple
import torch
import torch.nn as nn
from d2l.base.model import ClassifierModel
import d2l.base.function as d2l_F
from abc import abstractmethod
from .block import VGGBlock, NiNBlock, InceptionBlock, ResidualBlock, ResidualXBlock, DenseBlock, TransitionBlock, BatchNorm1d, BatchNorm2d

class SoftmaxClassifier(ClassifierModel):
    def __init__(self, 
                 num_features: int,
                 num_outputs: int, 
                 rng: torch.Generator = torch.Generator().manual_seed(42)) -> None:
        super().__init__()
        self.num_features = num_features
        self.num_outputs = num_outputs
        self.rng = rng

        self.W = nn.Parameter(torch.normal(0, 0.01, (num_features, num_outputs), generator=rng))
        self.b = nn.Parameter(torch.zeros(num_outputs))
    
    def loss(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        probs = d2l_F.softmax(y_hat) 
        correct_probs = probs[range(len(y_hat)), y] 
        return -torch.log(correct_probs).mean()
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = X.reshape(X.shape[0], -1)
        return X @ self.W + self.b
    
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        return self.forward(X).argmax(dim=1)

class SoftmaxClassifierLogSumExp(SoftmaxClassifier):
    def __init__(self, 
                 num_features: int,
                 num_outputs: int, 
                 rng: torch.Generator = torch.Generator().manual_seed(42)) -> None:
        super().__init__(num_features, num_outputs, rng)
        
    
    def loss(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        log_sum_exp = d2l_F.log_sum_exp(y_hat)
        correct_prob = y_hat[range(len(y_hat)), y]
        return (log_sum_exp - correct_prob).mean()

class SoftmaxClassifierTorch(ClassifierModel):
    def __init__(self, 
                 num_features: int,
                 num_outputs: int, 
                 rng: torch.Generator = torch.Generator().manual_seed(42)) -> None:
        super().__init__()
        self.num_features = num_features
        self.num_outputs = num_outputs
        self.rng = rng

        flatten = torch.nn.Flatten()
        linear = torch.nn.Linear(num_features, num_outputs)
        torch.nn.init.normal_(linear.weight, 0, 0.01, generator=self.rng)  
        torch.nn.init.zeros_(linear.bias)
        self.net = torch.nn.Sequential(flatten, linear)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.net(X)

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        return self.forward(X).argmax(dim=1)
    
    def loss(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.cross_entropy(y_hat, y)

class MLPClassifier(ClassifierModel):
    def __init__(self,
                 num_features: int,
                 num_outputs: int, 
                 num_hiddens: List[int],
                 rng: torch.Generator = torch.Generator().manual_seed(42)) -> None:
        super().__init__()
        self.num_features = num_features
        self.num_outputs = num_outputs
        self.num_hiddens = num_hiddens
        self.rng = rng

        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        layer_sizes = [num_features] + num_hiddens + [num_outputs]
        for i in range(len(layer_sizes) - 1):
            d, h = layer_sizes[i], layer_sizes[i + 1]
            W = nn.Parameter(torch.normal(0, 0.01, (d, h), generator=rng))
            b = nn.Parameter(torch.zeros(h))
            self.weights.append(W)
            self.biases.append(b)
            
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = X.reshape(X.shape[0], -1)
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            X = X @ W + b
            if i != len(self.weights) - 1:
                X = d2l_F.relu(X)
        return X
    
    def loss(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        probs = d2l_F.softmax(y_hat) 
        correct_probs = probs[range(len(y_hat)), y] 
        return -torch.log(correct_probs).mean()
    
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        return self.forward(X).argmax(dim=1)
        
class MLPClassifierTorch(ClassifierModel):
    def __init__(self, 
                 num_features: int,
                 num_outputs: int, 
                 num_hiddens: List[int],
                 rng: torch.Generator = torch.Generator().manual_seed(42)) -> None:
        super().__init__()
        self.num_features = num_features
        self.num_outputs = num_outputs
        self.num_hiddens = num_hiddens
        self.rng = rng

        self.net = self.make_net()
        
    def make_net(self) -> torch.nn.Module:
        layers: List[torch.nn.Module] = [torch.nn.Flatten()]  # Add flatten layer for image input
        input_size = self.num_features
        for hidden_size in self.num_hiddens:
            layers.append(torch.nn.Linear(input_size, hidden_size))
            layers.append(torch.nn.ReLU())
            input_size = hidden_size
        layers.append(torch.nn.Linear(input_size, self.num_outputs))
        net = torch.nn.Sequential(*layers)

        for layer in net:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.normal_(layer.weight, 0, 0.01, generator=self.rng)
                torch.nn.init.zeros_(layer.bias)
        return net
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.net(X)

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        return self.forward(X).argmax(dim=1)
    
    def loss(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.cross_entropy(y_hat, y, reduction='mean')
    
class MLPClassifierDropout(MLPClassifier):
    def __init__(self,
                 num_features: int,
                 num_outputs: int, 
                 num_hiddens: List[int],
                 dropouts: List[float],
                 rng: torch.Generator = torch.Generator().manual_seed(42)) -> None:
        self.dropouts = dropouts
        super().__init__(num_features, num_outputs, num_hiddens, rng)
        
    def dropout(self, X: torch.Tensor, drop_prob: float) -> torch.Tensor:
        if drop_prob <= 0.0 or drop_prob >= 1.0:
            return X
        mask = (torch.rand(X.shape, generator=self.rng) > drop_prob).float()
        return X * mask / (1.0 - drop_prob)
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = X.reshape(X.shape[0], -1)
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            X = X @ W + b
            if i != len(self.weights) - 1:
                X = d2l_F.relu(X)
                if self.training:
                    X = self.dropout(X, self.dropouts[i])
        return X
    
class MLPClassifierDropoutTorch(MLPClassifierTorch):
    def __init__(self, 
                 num_features: int,
                 num_outputs: int, 
                 num_hiddens: List[int],
                 dropouts: List[float],
                 rng: torch.Generator = torch.Generator().manual_seed(42)) -> None:
        self.dropouts = dropouts
        super().__init__(num_features=num_features,
                      num_outputs=num_outputs,
                      num_hiddens=num_hiddens,
                      rng=rng) 

    def make_net(self) -> torch.nn.Module:
        layers: List[torch.nn.Module] = [torch.nn.Flatten()]  # Add flatten layer for image input
        input_size = self.num_features
        for i, hidden_size in enumerate(self.num_hiddens):
            layers.append(torch.nn.Linear(input_size, hidden_size))
            layers.append(torch.nn.ReLU())
            if self.dropouts[i] > 0.0:
                layers.append(torch.nn.Dropout(p=self.dropouts[i]))
            input_size = hidden_size
        layers.append(torch.nn.Linear(input_size, self.num_outputs))
        net = torch.nn.Sequential(*layers)

        for layer in net:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.normal_(layer.weight, 0, 0.01, generator=self.rng)
                torch.nn.init.zeros_(layer.bias)
        return net
    
class ConvolutionalClassifierTorch(ClassifierModel):
    def __init__(self,
                 num_outputs: int,
                 rng: torch.Generator = torch.Generator().manual_seed(42)) -> None:
        super().__init__()
        self.num_outputs = num_outputs
        self.rng = rng

        self.net = self.make_net()
        
    @abstractmethod
    def make_net(self) -> torch.nn.Module:
        pass
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.net(X)

    def init(self, X_shape: Tuple[int, ...]) -> None:
        X = torch.randn(*X_shape, generator=self.rng)
        self.forward(X)
        assert isinstance(self.net, torch.nn.Sequential), "init only supports Sequential models."
        for layer in self.net:
            if isinstance(layer, torch.nn.Linear) or isinstance(layer, torch.nn.Conv2d):
                nn.init.xavier_uniform_(layer.weight, gain=1, generator=self.rng)
                
    def layer_summary(self, X_shape: Tuple[int, ...]) -> None:
        X = torch.randn(*X_shape, generator=self.rng)
        assert isinstance(self.net, torch.nn.Sequential), "layer_summary only supports Sequential models."
        for layer in self.net:
            print(f'{layer.__class__.__name__:>20}  input shape: {X.shape}')
            X = layer(X)
            print(f'{layer.__class__.__name__:>20}  output shape: {X.shape}')
        print(f'{"Total":>20}  output shape: {X.shape}')
        
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        return self.forward(X).argmax(dim=1)
    
    def loss(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.cross_entropy(y_hat, y, reduction='mean')

class LeNetClassifierTorch(ConvolutionalClassifierTorch):
    def __init__(self,
                num_outputs: int,
                rng: torch.Generator = torch.Generator().manual_seed(42)) -> None:
        self.num_outputs = num_outputs
        self.rng = rng
        
        super().__init__(num_outputs, rng)

    def make_net(self) -> torch.nn.Module:
        net = torch.nn.Sequential(
            nn.LazyConv2d(6, kernel_size=5, padding=2), nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.LazyConv2d(16, kernel_size=5), nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.LazyLinear(120), nn.Sigmoid(),
            nn.LazyLinear(84), nn.Sigmoid(),
            nn.LazyLinear(self.num_outputs)
        )
        return net
    
class AlexNetClassifierTorch(ConvolutionalClassifierTorch):
    def __init__(self,
                num_outputs: int,
                rng: torch.Generator = torch.Generator().manual_seed(42)) -> None:
        self.num_outputs = num_outputs
        self.rng = rng
        
        super().__init__(num_outputs, rng)

    def make_net(self) -> torch.nn.Module:
        net = torch.nn.Sequential(
            nn.LazyConv2d(32, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LazyConv2d(64, kernel_size=5, padding=2), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LazyConv2d(128, kernel_size=3, padding=1), nn.ReLU(),
            nn.LazyConv2d(256, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Flatten(),
            nn.LazyLinear(512), nn.ReLU(), # MODIFIED: 4096 -> 512
            nn.Dropout(p=0.5),
            nn.LazyLinear(512), nn.ReLU(), # MODIFIED: 4096 -> 512
            nn.Dropout(p=0.5),
            nn.LazyLinear(self.num_outputs)
        )
        return net

    
class VGGClassifier(ConvolutionalClassifierTorch):
    def __init__(self,
                num_outputs: int,
                rng: torch.Generator = torch.Generator().manual_seed(42)) -> None:
        self.num_outputs = num_outputs
        self.architecture = [(2, 128), (2, 128)]
        self.rng = rng
        
        super().__init__(num_outputs, rng)

    def make_net(self) -> torch.nn.Module:
        conv_blk : List[torch.nn.Module] = []
        for (num_convs, out_channels) in self.architecture:
            conv_blk.append(VGGBlock(num_convs, out_channels))
        net = torch.nn.Sequential(
            *conv_blk,
            nn.Flatten(),
            nn.LazyLinear(512), nn.ReLU(), # MODIFIED: 4096 -> 512
            nn.Dropout(p=0.5),
            nn.LazyLinear(512), nn.ReLU(), # MODIFIED: 4096 -> 512
            nn.Dropout(p=0.5),
            nn.LazyLinear(self.num_outputs)
        )
        return net  
    
    
class NiNClassifier(ConvolutionalClassifierTorch):
    def __init__(self,
                num_outputs: int,
                rng: torch.Generator = torch.Generator().manual_seed(42)) -> None:
        self.num_outputs = num_outputs
        self.rng = rng
        
        super().__init__(num_outputs, rng)

    def make_net(self) -> torch.nn.Module:
        net = torch.nn.Sequential(
            NiNBlock(64, kernel_size=5, stride=1, padding=2), # 256 -> 64
            nn.MaxPool2d(kernel_size=3, stride=2),
            NiNBlock(128, kernel_size=3, stride=1, padding=1), # 384 -> 128
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Dropout(p=0.5),
            NiNBlock(self.num_outputs, kernel_size=3, stride=1, padding=1), # MODIFIED
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        return net
        
class GoogLeNetClassifier(ConvolutionalClassifierTorch):
    def __init__(self,
                num_outputs: int,
                rng: torch.Generator = torch.Generator().manual_seed(42)) -> None:
        self.num_outputs = num_outputs
        self.rng = rng
        
        super().__init__(num_outputs, rng)

    def make_net(self) -> torch.nn.Module:
        net = torch.nn.Sequential(
            self.stem(),
            self.body(),
            self.head(self.num_outputs)
        )
        return net
    
    @classmethod
    def stem(cls) -> torch.nn.Module:
        return nn.Sequential(
            nn.LazyConv2d(16, kernel_size=7, stride=2, padding=3), nn.ReLU(), # 64 -> 16
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.LazyConv2d(16, kernel_size=1), nn.ReLU(), # 64 -> 16
            nn.LazyConv2d(48, kernel_size=3, padding=1), nn.ReLU(), # 192 -> 48
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
    @classmethod
    def body1(cls) -> torch.nn.Module:
        return nn.Sequential(
            InceptionBlock(16, 24, 32, 4, 8, 8),  # Original: (64, 96, 128, 16, 32, 32)
            InceptionBlock(32, 32, 48, 8, 24, 16), # Original: (128, 128, 192, 32, 96, 64)
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

    @classmethod
    def body2(cls) -> torch.nn.Module:
        return nn.Sequential(
            InceptionBlock(48, 24, 52, 4, 12, 16),  # Original: (192, 96, 208, 16, 48, 64)
            InceptionBlock(40, 28, 56, 6, 16, 16),  # Original: (160, 112, 224, 24, 64, 64)
            # Removed 3 Inception blocks for speed
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
    @classmethod
    def body3(cls) -> torch.nn.Module:
        return nn.Sequential(
            InceptionBlock(64, 40, 80, 8, 32, 32),  # Original: (256, 160, 320, 32, 128, 128)
            InceptionBlock(96, 48, 96, 12, 32, 32), # Original: (384, 192, 384, 48, 128, 128)
            nn.AdaptiveAvgPool2d((1, 1))
        )

    @classmethod
    def body(cls) -> torch.nn.Module:
        return nn.Sequential(
            cls.body1(),
            cls.body2(),
            cls.body3(),
            nn.Flatten()
        )
    
    @classmethod
    def head(cls, num_outputs: int) -> torch.nn.Module:
        return nn.Sequential(
            nn.LazyLinear(num_outputs)
        )
    
class LeNetBNClassifier(ConvolutionalClassifierTorch):
    def __init__(self,
                num_outputs: int,
                rng: torch.Generator = torch.Generator().manual_seed(42)) -> None:
        self.num_outputs = num_outputs
        self.rng = rng
        
        super().__init__(num_outputs, rng)

    def make_net(self) -> torch.nn.Module:
        net = torch.nn.Sequential(
            nn.LazyConv2d(6, kernel_size=5, padding=2), BatchNorm2d(6), nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.LazyConv2d(16, kernel_size=5), BatchNorm2d(16), nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.LazyLinear(120), BatchNorm1d(120), nn.Sigmoid(),
            nn.LazyLinear(84), BatchNorm1d(84), nn.Sigmoid(),
            nn.LazyLinear(self.num_outputs)
        )
        return net
    
class LeNetBNClassifierTorch(ConvolutionalClassifierTorch):
    def __init__(self,
                num_outputs: int,
                rng: torch.Generator = torch.Generator().manual_seed(42)) -> None:
        self.num_outputs = num_outputs
        self.rng = rng
        
        super().__init__(num_outputs, rng)

    def make_net(self) -> torch.nn.Module:
        net = torch.nn.Sequential(
            nn.LazyConv2d(6, kernel_size=5, padding=2), nn.LazyBatchNorm2d(), nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.LazyConv2d(16, kernel_size=5), nn.LazyBatchNorm2d(), nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.LazyLinear(120), nn.LazyBatchNorm1d(), nn.Sigmoid(),
            nn.LazyLinear(84), nn.LazyBatchNorm1d(), nn.Sigmoid(),
            nn.LazyLinear(self.num_outputs)
        )
        return net
    
class ResNetClassifier(ConvolutionalClassifierTorch):
    def __init__(self,
                num_outputs: int,
                rng: torch.Generator = torch.Generator().manual_seed(42)) -> None:
        self.num_outputs = num_outputs
        self.rng = rng
        
        super().__init__(num_outputs, rng)

    def make_net(self) -> torch.nn.Module:
        net = torch.nn.Sequential(
            nn.LazyConv2d(32, kernel_size=7, stride=2, padding=3), nn.LazyBatchNorm2d(), nn.ReLU(), # 64 -> 32
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ResidualBlock(32), # 64 -> 32
            ResidualBlock(64, stride=2), # 128 -> 64
            ResidualBlock(128, stride=2), # 256 -> 128
            ResidualBlock(256, stride=2), # 512 -> 256
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.LazyLinear(self.num_outputs)
        )
        return net
    
class ResNeXtClassifier(ConvolutionalClassifierTorch):
    def __init__(self,
                num_outputs: int,
                group_channels: int = 32,
                rng: torch.Generator = torch.Generator().manual_seed(42)) -> None:
        self.num_outputs = num_outputs
        self.group_channels = group_channels
        self.rng = rng
        
        super().__init__(num_outputs, rng)

    def make_net(self) -> torch.nn.Module:
        net = torch.nn.Sequential(
            nn.LazyConv2d(32, kernel_size=7, stride=2, padding=3), nn.LazyBatchNorm2d(), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ResidualXBlock(32, group_channels=self.group_channels),
            ResidualXBlock(64, stride=2, group_channels=self.group_channels),
            ResidualXBlock(128, stride=2, group_channels=self.group_channels), 
            ResidualXBlock(256, stride=2, group_channels=self.group_channels),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.LazyLinear(self.num_outputs)
        )
        return net
    
class DenseNetClassifier(ConvolutionalClassifierTorch):
    def __init__(self,
                num_outputs: int,
                rng: torch.Generator = torch.Generator().manual_seed(42)) -> None:
        self.num_outputs = num_outputs
        self.rng = rng
        
        super().__init__(num_outputs, rng)

    def make_net(self) -> torch.nn.Module:
        growth_rate = 16 # Original: 32
        num_convs_in_block = 2 # Original: 4
        
        net = torch.nn.Sequential(
            nn.LazyConv2d(32, kernel_size=7, stride=2, padding=3), nn.LazyBatchNorm2d(), nn.ReLU(), # 64 -> 32
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        
        # channels = 32
        net.add_module("DenseBlock_1", DenseBlock(num_convs=num_convs_in_block, growth_rate=growth_rate))
        # channels = 32 + 2 * 16 = 64
        net.add_module("TransitionBlock_1", TransitionBlock(out_channel=32)) # 64 // 2 = 32
        
        # channels = 32
        net.add_module("DenseBlock_2", DenseBlock(num_convs=num_convs_in_block, growth_rate=growth_rate))
        # channels = 32 + 2 * 16 = 64
        net.add_module("TransitionBlock_2", TransitionBlock(out_channel=32)) # 64 // 2 = 32
        
        # channels = 32
        net.add_module("DenseBlock_3", DenseBlock(num_convs=num_convs_in_block, growth_rate=growth_rate))
        # channels = 32 + 2 * 16 = 64
        
        net.add_module("Final_BN_ReLU", nn.Sequential(nn.LazyBatchNorm2d(), nn.ReLU()))
        net.add_module("GlobalAvgPool", nn.AdaptiveAvgPool2d((1, 1)))
        net.add_module("Flatten", nn.Flatten())
        net.add_module("Linear", nn.LazyLinear(self.num_outputs))
        return net
