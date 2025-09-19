#!/usr/bin/env python3
"""测试设备迁移功能"""

import torch
from torch import nn
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from d2l.model import ModelTorch

def test_device_migration():
    """测试模型设备迁移"""
    print("=== 测试设备迁移功能 ===\n")
    
    # 检查可用设备
    device_cpu = torch.device('cpu')
    if torch.cuda.is_available():
        device_gpu = torch.device('cuda')
        print(f"✅ 检测到CUDA设备: {torch.cuda.get_device_name()}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device_gpu = torch.device('mps')
        print("✅ 检测到MPS设备 (Apple Silicon)")
    else:
        device_gpu = device_cpu
        print("⚠️  仅使用CPU设备")
    
    # 创建简单模型
    net = nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 1)
    )
    
    # 创建优化器
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()
    
    # 创建模型
    model = ModelTorch(net, loss_fn, optimizer)
    
    # 创建一些测试数据
    X = torch.randn(32, 4)
    y = torch.randn(32, 1)
    
    print(f"1. 初始设备状态:")
    print(f"   网络参数设备: {next(model.net.parameters()).device}")
    print(f"   输入数据设备: {X.device}")
    
    # 进行一次训练步骤以初始化优化器状态
    print(f"\n2. 初始化优化器状态...")
    y_pred = model.forward(X)
    loss = model.loss(y_pred, y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    print(f"   优化器状态已初始化")
    
    # 测试设备迁移
    if device_gpu != device_cpu:
        print(f"\n3. 迁移到GPU设备: {device_gpu}")
        
        # 使用你的方法迁移
        model.to_device(device_gpu)
        
        # 检查迁移结果
        print(f"   网络参数设备: {next(model.net.parameters()).device}")
        
        # 检查优化器状态
        has_optimizer_state = any(model.optimizer.state.values())
        if has_optimizer_state:
            first_param_id = next(iter(model.optimizer.state.keys()))
            first_state = model.optimizer.state[first_param_id]
            if 'momentum_buffer' in first_state:
                print(f"   优化器动量设备: {first_state['momentum_buffer'].device}")
        
        # 测试前向传播（数据也需要迁移）
        X_gpu = X.to(device_gpu)
        y_gpu = y.to(device_gpu)
        
        with torch.no_grad():
            y_pred_gpu = model.forward(X_gpu)
            print(f"   GPU前向传播输出设备: {y_pred_gpu.device}")
            print(f"   ✅ GPU设备迁移成功!")
        
        # 迁移回CPU
        print(f"\n4. 迁移回CPU设备")
        model.to_device(device_cpu)
        print(f"   网络参数设备: {next(model.net.parameters()).device}")
        print(f"   ✅ CPU设备迁移成功!")
    
    else:
        print(f"\n3. 跳过GPU测试（仅CPU环境）")
    
    print(f"\n=== 测试完成 ===")

if __name__ == "__main__":
    test_device_migration()