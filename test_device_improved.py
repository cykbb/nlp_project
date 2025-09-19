#!/usr/bin/env python3
"""测试改进后的设备迁移功能"""

import torch
from torch import nn
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from d2l.model import ModelTorch

def test_improved_device_migration():
    """测试改进后的设备迁移"""
    print("=== 测试改进后的设备迁移功能 ===\n")
    
    # 创建模型
    net = nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(), 
        nn.Linear(8, 1)
    )
    
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    loss_fn = nn.MSELoss()
    model = ModelTorch(net, loss_fn, optimizer)
    
    # 创建测试数据
    X = torch.randn(16, 4)
    y = torch.randn(16, 1)
    
    print("1. 初始状态 (CPU):")
    print(f"   网络设备: {next(model.net.parameters()).device}")
    
    # 进行几步训练以建立优化器状态
    print("\n2. 建立优化器状态 (momentum buffers)...")
    for _ in range(3):
        y_pred = model.forward(X)
        loss = model.loss(y_pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    print(f"   优化器状态条目数: {len(model.optimizer.state)}")
    
    # 测试不同的迁移方法
    print("\n3. 测试链式调用方法:")
    
    # 测试MPS迁移
    model_mps = model.mps()
    print(f"   MPS迁移后设备: {next(model_mps.net.parameters()).device}")
    
    # 测试CPU迁移
    model_cpu = model.cpu()
    print(f"   CPU迁移后设备: {next(model_cpu.net.parameters()).device}")
    
    # 测试CUDA迁移
    model_cuda = model.cuda()
    print(f"   CUDA迁移尝试后设备: {next(model_cuda.net.parameters()).device}")
    
    print("\n4. 验证优化器状态迁移:")
    
    # 迁移到可用的GPU设备进行验证
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    model.to_device(device)
    
    # 检查优化器状态是否正确迁移
    optimizer_states_migrated = 0
    for param_state in model.optimizer.state.values():
        for key, value in param_state.items():
            if isinstance(value, torch.Tensor):
                if value.device == device:
                    optimizer_states_migrated += 1
                print(f"   优化器状态 '{key}' 设备: {value.device}")
    
    print(f"   成功迁移的优化器状态数量: {optimizer_states_migrated}")
    
    # 测试迁移后的训练
    print("\n5. 测试迁移后训练:")
    X_device = X.to(device)
    y_device = y.to(device)
    
    try:
        y_pred = model.forward(X_device)
        loss = model.loss(y_pred, y_device)
        loss.backward()
        model.optimizer.step()
        model.optimizer.zero_grad()
        print("   ✅ 迁移后训练成功!")
        print(f"   训练损失: {loss.item():.6f}")
    except Exception as e:
        print(f"   ❌ 迁移后训练失败: {e}")
    
    print("\n=== 测试完成 ===")

def test_edge_cases():
    """测试边界情况"""
    print("\n=== 测试边界情况 ===")
    
    # 1. 空优化器状态的模型
    net = nn.Linear(2, 1)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()
    model = ModelTorch(net, loss_fn, optimizer)
    
    print("1. 测试空优化器状态迁移:")
    try:
        model.to_device(torch.device('cpu'))
        print("   ✅ 空优化器状态迁移成功")
    except Exception as e:
        print(f"   ❌ 空优化器状态迁移失败: {e}")
    
    # 2. 测试重复迁移
    print("\n2. 测试重复迁移:")
    try:
        model.cpu().cpu().mps().cpu()
        print("   ✅ 重复迁移成功")
    except Exception as e:
        print(f"   ❌ 重复迁移失败: {e}")

if __name__ == "__main__":
    test_improved_device_migration()
    test_edge_cases()