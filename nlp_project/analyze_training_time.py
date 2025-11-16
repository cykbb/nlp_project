"""
Analyze training time from metrics file.
"""
import json
from pathlib import Path

def analyze_training_time(metrics_file: Path):
    """Analyze and display training time statistics."""
    if not metrics_file.exists():
        print(f"Metrics file not found: {metrics_file}")
        return
    
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
    
    print(f"\n{'=' * 70}")
    print(f"Training Time Analysis - {metrics_file.parent.name}")
    print(f"{'=' * 70}\n")
    
    for model_name, epochs in metrics.items():
        print(f"{model_name.upper()} Model:")
        print(f"{'-' * 70}")
        
        # Check if epoch_time field exists
        if epochs and 'epoch_time' in epochs[0]:
            total_time = sum(epoch['epoch_time'] for epoch in epochs)
            avg_time = total_time / len(epochs)
            min_time = min(epoch['epoch_time'] for epoch in epochs)
            max_time = max(epoch['epoch_time'] for epoch in epochs)
            
            print(f"  Total epochs:         {len(epochs)}")
            print(f"  Total training time:  {total_time:.1f}s ({total_time/60:.1f}m)")
            print(f"  Average time/epoch:   {avg_time:.1f}s")
            print(f"  Min time/epoch:       {min_time:.1f}s")
            print(f"  Max time/epoch:       {max_time:.1f}s")
            
            # Find best epoch
            best_epoch_idx = max(range(len(epochs)), key=lambda i: epochs[i]['val_accuracy'])
            best_epoch = epochs[best_epoch_idx]
            print(f"  Best epoch:           {best_epoch['epoch']} "
                  f"(acc: {best_epoch['val_accuracy']:.4f}, time: {best_epoch['epoch_time']:.1f}s)")
            
            # Show per-epoch breakdown
            print(f"\n  Epoch-by-Epoch Breakdown:")
            print(f"  {'Epoch':>6} | {'Time (s)':>10} | {'Val Acc':>8} | {'Val Loss':>8}")
            print(f"  {'-' * 6}-+-{'-' * 10}-+-{'-' * 8}-+-{'-' * 8}")
            for epoch in epochs:
                marker = " *" if epoch['epoch'] == best_epoch['epoch'] else ""
                print(f"  {epoch['epoch']:>6} | {epoch['epoch_time']:>10.1f} | "
                      f"{epoch['val_accuracy']:>8.4f} | {epoch['val_loss']:>8.4f}{marker}")
        else:
            print(f"  No timing information available (old format)")
            print(f"  Total epochs: {len(epochs)}")
        
        print()

if __name__ == "__main__":
    # Check exp3 first (latest with timing)
    exp3_metrics = Path(__file__).parent / "exp3" / "training_metrics.json"
    if exp3_metrics.exists():
        analyze_training_time(exp3_metrics)
    
    # Also check exp1 for comparison
    exp1_metrics = Path(__file__).parent / "exp1" / "training_metrics.json"
    if exp1_metrics.exists():
        analyze_training_time(exp1_metrics)
