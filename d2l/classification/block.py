from typing import List
import torch
import torch.nn as nn

class VGGBlock(nn.Module):
    def __init__(self, 
                 num_convs: int,
                out_channels: int) -> None:
        super().__init__()
        layers: List[torch.nn.Module] = []
        for _ in range(num_convs):
            layers.append(nn.LazyConv2d(out_channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.block = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)
    
class NiNBlock(nn.Module):
    def __init__(self,
                 out_channels: int,
                 kernel_size: int,
                 stride: int,
                 padding: int) -> None:
        super().__init__()
        self.nin = nn.Sequential(
            nn.LazyConv2d(out_channels, kernel_size, stride, padding), nn.ReLU(),
            nn.LazyConv2d(out_channels, kernel_size=1), nn.ReLU(),
            nn.LazyConv2d(out_channels, kernel_size=1), nn.ReLU()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.nin(x)
    
class InceptionBlock(nn.Module):
    def __init__(self,
                    c1_out: int, 
                    c2_out1x1: int,
                    c2_out3x3: int,
                    c3_out1x1: int,
                    c3_out5x5: int,
                    c4_out: int) -> None:
        
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.LazyConv2d(c1_out, kernel_size=1),
            nn.ReLU()
        )
        self.branch2 = nn.Sequential(
            nn.LazyConv2d(c2_out1x1, kernel_size=1),
            nn.ReLU(),
            nn.LazyConv2d(c2_out3x3, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.branch3 = nn.Sequential(
            nn.LazyConv2d(c3_out1x1, kernel_size=1),
            nn.ReLU(),
            nn.LazyConv2d(c3_out5x5, kernel_size=5, padding=2),
            nn.ReLU()
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.LazyConv2d(c4_out, kernel_size=1),
            nn.ReLU()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out4 = self.branch4(x)
        return torch.cat([out1, out2, out3, out4], dim=1)
    
    
     
class BatchNorm1d(nn.Module):
    def __init__(self, num_features: int, momentum: float = 0.9, eps: float = 1e-5) -> None:
        super().__init__()
        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps
        self.shape = (1, num_features)
        
        self.gamma = nn.Parameter(torch.ones(self.shape))
        self.beta = nn.Parameter(torch.zeros(self.shape))
        
        self.moving_mean = torch.zeros(self.shape)
        self.moving_var = torch.ones(self.shape)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if self.training:
            batch_mean = X.mean(dim=0, keepdim=True)
            batch_var = X.var(dim=0, unbiased=False, keepdim=True)
            X_hat = (X - batch_mean) / torch.sqrt(batch_var + self.eps)
            with torch.no_grad():
                self.moving_mean.mul_(self.momentum).add_((1.0 - self.momentum) * batch_mean)
                self.moving_var.mul_(self.momentum).add_((1.0 - self.momentum) * batch_var)
        else:
            X_hat = (X - self.moving_mean) / torch.sqrt(self.moving_var + self.eps)
        Y = self.gamma * X_hat + self.beta
        return Y
        
class BatchNorm2d(nn.Module):
    def __init__(self, num_features: int, momentum: float = 0.9, eps: float = 1e-5) -> None:
        super().__init__()
        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps
        self.shape = (1, num_features, 1, 1)
        
        self.gamma = nn.Parameter(torch.ones(self.shape))
        self.beta = nn.Parameter(torch.zeros(self.shape))

        self.moving_mean = torch.zeros(self.shape)
        self.moving_var = torch.ones(self.shape)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if self.training:
            batch_mean = X.mean(dim=(0, 2, 3), keepdim=True)
            batch_var = X.var(dim=(0, 2, 3), keepdim=True, unbiased=False)
            X_hat = (X - batch_mean) / torch.sqrt(batch_var + self.eps)
            with torch.no_grad():
                self.moving_mean.mul_(self.momentum).add_((1.0 - self.momentum) * batch_mean)
                self.moving_var.mul_(self.momentum).add_((1.0 - self.momentum) * batch_var)
        else:
            X_hat = (X - self.moving_mean) / torch.sqrt(self.moving_var + self.eps)
        Y = self.gamma * X_hat + self.beta
        return Y
        
class LayerNorm(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        mean = X.mean(dim=1, keepdim=True)
        var = X.var(dim=1, keepdim=True, unbiased=False)
        X_hat = (X - mean) / torch.sqrt(var + self.eps)
        Y = self.gamma * X_hat + self.beta
        return Y
    
class ResidualBlock(nn.Module):
    def __init__(self, 
                 num_channels: int, 
                 stride: int = 1) -> None:
    
        super().__init__()
        self.stride = stride
        self.conv1 = nn.LazyConv2d(num_channels, kernel_size=3, padding=1, stride=stride)
        self.bn1 = nn.LazyBatchNorm2d()
        self.conv2 = nn.LazyConv2d(num_channels, kernel_size=3, padding=1)
        self.bn2 = nn.LazyBatchNorm2d()
        if stride != 1:
            self.downsample = nn.LazyConv2d(num_channels, kernel_size=1, stride=stride)
            self.downsample_bn = nn.LazyBatchNorm2d()
            
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        Y = torch.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.stride != 1:
            X = self.downsample_bn(self.downsample(X))
        Y += X
        return torch.relu(Y)
    
class Conv2dGroup(nn.Module):
    def __init__(self, 
                 out_channels: int, 
                 kernel_size: int, 
                 stride: int = 1, 
                 padding: int = 0, 
                 groups: int = 1) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        
        self.group_width = out_channels // groups
        convs = []
        for gid in range(groups):
            if (gid + 1) * self.group_width >= out_channels:
                convs.append(nn.LazyConv2d(self.out_channels - gid * self.group_width, kernel_size, stride, padding))
                break
            convs.append(nn.LazyConv2d(self.group_width, kernel_size, stride, padding))
        self.convs = nn.ModuleList(convs)
        
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X_split = torch.chunk(X, self.groups, dim=1)
        Y_split = [conv(x) for conv, x in zip(self.convs, X_split)]
        Y = torch.cat(Y_split, dim=1)
        return Y
        
class ResidualXBlock(nn.Module):
    def __init__(self, 
                 num_channels: int, 
                 bottleneck_multiplier: float = 1.0,
                 group_channels: int = 1,
                 stride: int = 1, 
                 ) -> None:
    
        super().__init__()
        self.stride = stride
        self.bottleneck_multiplier = bottleneck_multiplier
        self.group_channels = group_channels
        self.bottleneck_channels = int(num_channels * bottleneck_multiplier)
        self.num_groups = self.bottleneck_channels // self.group_channels
        self.use_downsample = stride != 1
        self.conv1 = nn.LazyConv2d(self.bottleneck_channels, kernel_size=1)
        self.bn1 = nn.LazyBatchNorm2d()
        # self.conv2 = Conv2dGroup(self.bottleneck_channels, kernel_size=3, padding=1, stride=stride, groups=self.num_groups)
        self.conv2 = nn.LazyConv2d(self.bottleneck_channels, kernel_size=3, padding=1, stride=stride, groups=self.num_groups)
        self.bn2 = nn.LazyBatchNorm2d()
        self.conv3 = nn.LazyConv2d(num_channels, kernel_size=1)
        self.bn3 = nn.LazyBatchNorm2d()
        if self.use_downsample:
            self.downsample = nn.LazyConv2d(num_channels, kernel_size=1, stride=stride)
            self.downsample_bn = nn.LazyBatchNorm2d()
        
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        Y = torch.relu(self.bn1(self.conv1(X)))
        Y = torch.relu(self.bn2(self.conv2(Y)))
        Y = self.bn3(self.conv3(Y))
        if self.use_downsample:
            X = self.downsample_bn(self.downsample(X))
        Y += X
        return torch.relu(Y)
    
    
class DenseBlock(nn.Module):
    def __init__(self,
                 num_convs: int,
                 growth_rate: int) -> None:
        super().__init__()
        self.net = nn.ModuleList()
        for _ in range(num_convs):
            self.net.append(self._make_conv(growth_rate))
            
    def _make_conv(self, growth_rate: int) -> torch.nn.Module:
        return nn.Sequential(
            nn.LazyBatchNorm2d(),
            nn.ReLU(),
            nn.LazyConv2d(4 * growth_rate, kernel_size=1),
            nn.LazyBatchNorm2d(),
            nn.ReLU(),
            nn.LazyConv2d(growth_rate, kernel_size=3, padding=1)
        )
        
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        for conv in self.net:
            Y = conv(X)
            X = torch.cat([X, Y], dim=1)
        return X
    
class TransitionBlock(nn.Module):
    def __init__(self,
                 out_channel: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LazyBatchNorm2d(),
            nn.ReLU(),
            nn.LazyConv2d(out_channel, kernel_size=1),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.net(X)