"""
Evaluate trained model on the held-out test set.
Run this AFTER training completes to get final, unbiased performance metrics.

Usage:
    python test_data_eval.py
"""
import torch
import numpy as np
from tqdm import tqdm
import os
import json
import matplotlib.pyplot as plt
from modules import get_model
from dataset import get_data_loaders
from config import Config


def calculate_dice_per_class(pred, target, num_classes=6):
    """
    Calculate Dice score for each class.
    """
    if pred.dim() == 5:  # (B, C, D, H, W)
        pred = torch.argmax(pred, dim=1)
    if target.dim() == 5:  # (B, 1, D, H, W)
        target = target.squeeze(1)
    
    dice_scores = []
    for class_id in range(num_classes):
        pred_class = (pred == class_id)
        target_class = (target == class_id)
        intersection = torch.sum(pred_class & target_class).float()
        union = torch.sum(pred_class).float() + torch.sum(target_class).float()
        dice_scores.append((2.0 * intersection / (union + 1e-8)).item() if union > 0 else float('nan'))
    
    return dice_scores


def evaluate_test_set(checkpoint_path='./checkpoints/best_model.pth'):
    """
    Evaluate model on test set.
    """
    print("\n" + "="*70)
    print("FINAL TEST SET EVALUATION")
    print("="*70)
    
    if not os.path.exists(checkpoint_path):
        print(f"\n❌ ERROR: Checkpoint not found at {checkpoint_path}")
        print("   Please train the model first!")
        return None
    
    device = torch.device(Config.DEVICE)
    print(f"\n🚀 Using device: {device}")
    
    # Load model
    print("\n🏗️  Loading model...")
    model = get_model().to(device)
    
    # Load checkpoint safely
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)

        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        checkpoint_epoch = checkpoint.get('epoch', 'unknown')
        best_val_dice = checkpoint.get('mean_val_dice', checkpoint.get('best_dice', 'unknown'))
        
        print(f"✅ Loaded checkpoint from epoch {checkpoint_epoch}")
        print(f"   Best validation Dice: {best_val_dice:.4f}" if isinstance(best_val_dice, float) else f"   Best validation Dice: {best_val_dice}")
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        return None
    
    model.eval()
    
    # Load test data
    print("\n📊 Loading test data...")
    _, _, test_loader = get_data_loaders()
    print(f"   Test set size: {len(test_loader.dataset)} samples")
    
    print("\n🔍 Evaluating on test set...")
    
    class_names = ['Background', 'Body', 'Bone', 'Bladder', 'Rectum', 'Prostate']
    all_dice_scores = {name: [] for name in class_names}
    
    with torch.no_grad():
        for batch_idx, (mri, labels) in enumerate(tqdm(test_loader, desc="Testing")):
            mri, labels = mri.to(device), labels.to(device)
            outputs = model(mri)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            preds = torch.argmax(outputs, dim=1)
            dice_scores = calculate_dice_per_class(preds, labels, num_classes=6)
            for idx, name in enumerate(class_names):
                if not np.isnan(dice_scores[idx]):
                    all_dice_scores[name].append(dice_scores[idx])
    
    # Print results
    print("\n" + "="*70)
    print("TEST SET RESULTS")
    print("="*70)
    
    mean_dice_all = []
    std_dice_all = []
    for name in class_names:
        if len(all_dice_scores[name]) > 0:
            mean_dice = np.mean(all_dice_scores[name])
            std_dice = np.std(all_dice_scores[name])
            min_dice = np.min(all_dice_scores[name])
            max_dice = np.max(all_dice_scores[name])
            print(f"  {name:12s}: {mean_dice:.4f} ± {std_dice:.4f}  [range: {min_dice:.4f} - {max_dice:.4f}]")
            mean_dice_all.append(mean_dice)
            std_dice_all.append(std_dice)
        else:
            print(f"  {name:12s}: N/A (not present in test set)")
    
    overall_mean = np.mean(mean_dice_all[1:])  # excluding background
    overall_mean_all = np.mean(mean_dice_all)
    
    print(f"\n{'='*70}")
    print(f"Mean Dice (all classes):           {overall_mean_all:.4f}")
    print(f"Mean Dice (excluding background):  {overall_mean:.4f}")
    print(f"{'='*70}")
    
    # Prostate performance
    if len(all_dice_scores['Prostate']) > 0:
        prostate_mean = np.mean(all_dice_scores['Prostate'])
        prostate_std = np.std(all_dice_scores['Prostate'])
        prostate_min = np.min(all_dice_scores['Prostate'])
        prostate_max = np.max(all_dice_scores['Prostate'])
        
        print(f"\n🎯 TARGET ORGAN (Prostate):")
        print(f"   Mean:  {prostate_mean:.4f}")
        print(f"   Std:   {prostate_std:.4f}")
        print(f"   Range: [{prostate_min:.4f}, {prostate_max:.4f}]")
        
        print(f"\n   Performance Assessment:")
        if prostate_mean >= 0.70:
            print(f"   ✅ EXCELLENT performance! (Dice ≥ 0.70)")
        elif prostate_mean >= 0.60:
            print(f"   ✅ GOOD performance! (Dice ≥ 0.60)")
        elif prostate_mean >= 0.50:
            print(f"   ✅ ACCEPTABLE for this difficult task (Dice ≥ 0.50)")
        else:
            print(f"   ⚠️  Room for improvement (Dice < 0.50)")
    else:
        print("   ❌ No prostate samples found in test set")
    
    # Compare with validation
    if isinstance(best_val_dice, float):
        print(f"\n📊 Validation vs Test Comparison:")
        print(f"   Best Validation Dice: {best_val_dice:.4f}")
        print(f"   Test Dice:            {overall_mean:.4f}")
        diff = overall_mean - best_val_dice
        if abs(diff) < 0.03:
            print(f"   ✅ Consistent performance (diff: {diff:+.4f})")
        elif diff < -0.05:
            print(f"   ⚠️  Validation was optimistic (diff: {diff:+.4f})")
        else:
            print(f"   ✅ Test performance within expected range")
    
    # Save results
    results = {
        'checkpoint_epoch': checkpoint_epoch,
        'validation_dice': best_val_dice if isinstance(best_val_dice, float) else None,
        'test_dice_per_class': {
            name: {
                'mean': np.mean(all_dice_scores[name]) if len(all_dice_scores[name]) > 0 else None,
                'std': np.std(all_dice_scores[name]) if len(all_dice_scores[name]) > 0 else None,
                'min': np.min(all_dice_scores[name]) if len(all_dice_scores[name]) > 0 else None,
                'max': np.max(all_dice_scores[name]) if len(all_dice_scores[name]) > 0 else None,
            }
            for name in class_names
        },
        'overall_mean_dice': overall_mean,
        'overall_mean_dice_with_bg': overall_mean_all,
    }
    
    os.makedirs(Config.RESULTS_DIR, exist_ok=True)
    results_path = os.path.join(Config.RESULTS_DIR, 'test_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\n💾 Results saved to: {results_path}")
    
    create_results_visualization(all_dice_scores, class_names)
    
    print("="*70 + "\n")
    return results


def create_results_visualization(all_dice_scores, class_names):
    """Create a bar chart of test results"""
    means, stds, names = [], [], []
    for name in class_names:
        if len(all_dice_scores[name]) > 0:
            means.append(np.mean(all_dice_scores[name]))
            stds.append(np.std(all_dice_scores[name]))
            names.append(name)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(names))
    bars = ax.bar(x, means, yerr=stds, capsize=5, alpha=0.8, 
                   color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'][:len(names)])
    
    if 'Prostate' in names:
        prostate_idx = names.index('Prostate')
        bars[prostate_idx].set_color('#d62728')
        bars[prostate_idx].set_alpha(1.0)
        bars[prostate_idx].set_edgecolor('black')
        bars[prostate_idx].set_linewidth(2)
    
    ax.set_xlabel('Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('Dice Score', fontsize=12, fontweight='bold')
    ax.set_title('Test Set Performance - Per Class Dice Scores', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)
    
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + 0.02, f'{m:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    fig_path = os.path.join(Config.RESULTS_DIR, 'test_results_bar_chart.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"📊 Visualization saved to: {fig_path}")
    plt.close()


if __name__ == "__main__":
    print("\n" + "="*70)
    print("TEST EVALUATION SCRIPT")
    print("="*70)
    Config.print_config()
    
    results = evaluate_test_set()
    
    if results:
        print("\n✅ Test evaluation completed successfully!")
    else:
        print("\n❌ Test evaluation failed!")
