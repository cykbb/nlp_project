"""
Compare training time across different models.
"""
import json
from pathlib import Path
from typing import Dict, List

def format_time(seconds: float) -> str:
    """Format seconds to readable time string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m ({seconds:.0f}s)"
    else:
        hours = seconds / 3600
        minutes = (seconds % 3600) / 60
        return f"{hours:.1f}h {minutes:.0f}m"

def analyze_model_training_time(model_name: str, epochs: List[Dict]) -> Dict:
    """Analyze training time for a single model."""
    if not epochs or 'epoch_time' not in epochs[0]:
        return {
            'model': model_name,
            'has_timing': False
        }
    
    # Calculate time statistics
    total_time = sum(epoch['epoch_time'] for epoch in epochs)
    avg_time = total_time / len(epochs)
    min_time = min(epoch['epoch_time'] for epoch in epochs)
    max_time = max(epoch['epoch_time'] for epoch in epochs)
    
    # Find best epoch
    best_epoch_idx = max(range(len(epochs)), key=lambda i: epochs[i]['val_accuracy'])
    best_epoch = epochs[best_epoch_idx]
    time_to_best = sum(epoch['epoch_time'] for epoch in epochs[:best_epoch_idx + 1])
    
    return {
        'model': model_name,
        'has_timing': True,
        'num_epochs': len(epochs),
        'total_time': total_time,
        'avg_time_per_epoch': avg_time,
        'min_time_per_epoch': min_time,
        'max_time_per_epoch': max_time,
        'best_epoch': best_epoch['epoch'],
        'best_val_acc': best_epoch['val_accuracy'],
        'time_to_best': time_to_best,
        'epochs_data': epochs
    }

def print_comparison_table(exp_path: Path):
    """Print training time comparison table."""
    metrics_file = exp_path / "training_metrics.json"
    
    if not metrics_file.exists():
        print(f"Metrics file not found: {metrics_file}")
        return
    
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
    
    # Analyze all models
    results = {}
    for model_name, epochs in metrics.items():
        results[model_name] = analyze_model_training_time(model_name, epochs)
    
    # Check if timing data is available
    if not any(r['has_timing'] for r in results.values()):
        print(f"\n⚠ No timing information available in {exp_path.name}")
        return
    
    print(f"\n{'=' * 100}")
    print(f"Training Time Comparison - {exp_path.name}")
    print(f"{'=' * 100}\n")
    
    # Table 1: Overview
    print("Table 1: Training Time Overview")
    print(f"{'-' * 100}")
    print(f"{'Model':<8} | {'Total Epochs':<12} | {'Total Time':<20} | {'Avg Time/Epoch':<20} | {'Min/Max Time':<20}")
    print(f"{'-' * 100}")
    
    for model_name in sorted(results.keys()):
        r = results[model_name]
        if not r['has_timing']:
            print(f"{model_name.upper():<8} | No timing data available")
            continue
        
        print(f"{model_name.upper():<8} | "
              f"{r['num_epochs']:<12} | "
              f"{format_time(r['total_time']):<20} | "
              f"{format_time(r['avg_time_per_epoch']):<20} | "
              f"{format_time(r['min_time_per_epoch'])} / {format_time(r['max_time_per_epoch'])}")
    
    print()
    
    # Table 2: Best Model Performance
    print("\nTable 2: Time to Best Model")
    print(f"{'-' * 100}")
    print(f"{'Model':<8} | {'Best Epoch':<12} | {'Best Val Acc':<15} | {'Time to Best':<20} | {'% of Total Time':<20}")
    print(f"{'-' * 100}")
    
    for model_name in sorted(results.keys()):
        r = results[model_name]
        if not r['has_timing']:
            continue
        
        pct_time = (r['time_to_best'] / r['total_time']) * 100
        
        print(f"{model_name.upper():<8} | "
              f"{r['best_epoch']:<12} | "
              f"{r['best_val_acc']:.4f} ({r['best_val_acc']*100:.2f}%){'':<1} | "
              f"{format_time(r['time_to_best']):<20} | "
              f"{pct_time:.1f}%")
    
    print()
    
    # Table 3: Detailed Epoch Breakdown
    print("\nTable 3: Detailed Epoch-by-Epoch Breakdown\n")
    
    for model_name in sorted(results.keys()):
        r = results[model_name]
        if not r['has_timing']:
            continue
        
        print(f"{model_name.upper()} Model:")
        print(f"{'-' * 100}")
        print(f"{'Epoch':<8} | {'Time':<12} | {'Cumulative':<15} | {'Train Loss':<12} | {'Val Loss':<12} | {'Val Acc':<12}")
        print(f"{'-' * 100}")
        
        cumulative_time = 0
        for epoch in r['epochs_data']:
            cumulative_time += epoch['epoch_time']
            marker = " *" if epoch['epoch'] == r['best_epoch'] else ""
            
            print(f"{epoch['epoch']:<8} | "
                  f"{format_time(epoch['epoch_time']):<12} | "
                  f"{format_time(cumulative_time):<15} | "
                  f"{epoch['train_loss']:.4f}{'':<6} | "
                  f"{epoch['val_loss']:.4f}{'':<6} | "
                  f"{epoch['val_accuracy']:.4f}{marker}")
        
        print()
    
    # Speed comparison
    print("\nTable 4: Relative Speed Comparison (vs fastest)")
    print(f"{'-' * 100}")
    print(f"{'Model':<8} | {'Avg Time/Epoch':<20} | {'Speedup Factor':<20} | {'Total Time':<20}")
    print(f"{'-' * 100}")
    
    # Find fastest model
    min_avg_time = min(r['avg_time_per_epoch'] for r in results.values() if r['has_timing'])
    
    for model_name in sorted(results.keys()):
        r = results[model_name]
        if not r['has_timing']:
            continue
        
        speedup = r['avg_time_per_epoch'] / min_avg_time
        
        print(f"{model_name.upper():<8} | "
              f"{format_time(r['avg_time_per_epoch']):<20} | "
              f"{speedup:.2f}x{' (fastest)' if speedup == 1.0 else '':<13} | "
              f"{format_time(r['total_time'])}")
    
    print()

def main():
    """Main function to compare training times."""
    base_dir = Path(__file__).parent
    
    # Check exp3 (latest with timing)
    exp3_path = base_dir / "exp3"
    if exp3_path.exists() and (exp3_path / "training_metrics.json").exists():
        print_comparison_table(exp3_path)
    else:
        print(f"⚠ exp3 not found or no metrics available")
    
    # Also check exp1 if needed
    exp1_path = base_dir / "exp1"
    if exp1_path.exists() and (exp1_path / "training_metrics.json").exists():
        print_comparison_table(exp1_path)

if __name__ == "__main__":
    main()
