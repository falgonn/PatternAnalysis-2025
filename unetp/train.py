# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
import os
import time

# MONAI imports
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete

from config import Config
from modules import get_model
from dataset import get_data_loaders

# TensorBoard
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("⚠️  TensorBoard not available")


def calculate_dice_per_class(predictions, targets, num_classes):
    """
    Calculate Dice coefficient for each class
    
    Args:
        predictions: Model predictions (B, C, D, H, W)
        targets: Ground truth labels (B, D, H, W)
        num_classes: Number of classes
    
    Returns:
        dice_scores: Dice score for each class
    """
    # Convert predictions to class labels
    preds = torch.argmax(predictions, dim=1)  # (B, D, H, W)
    
    dice_scores = []
    for class_idx in range(num_classes):
        pred_class = (preds == class_idx).float()
        target_class = (targets == class_idx).float()
        
        intersection = (pred_class * target_class).sum()
        union = pred_class.sum() + target_class.sum()
        
        if union > 0:
            dice = (2.0 * intersection) / (union + 1e-8)
        else:
            dice = torch.tensor(1.0)  # Perfect score if both empty
        
        dice_scores.append(dice.item())
    
    return dice_scores


def train_epoch(model, loader, criterion, optimizer, scaler, device, epoch):
    """Train for one epoch"""
    model.train()
    
    running_loss = 0.0
    all_dice_scores = [[] for _ in range(Config.NUM_CLASSES)]
    
    pbar = tqdm(loader, desc=f'Epoch {epoch} [Train]')
    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision training
        with autocast(enabled=Config.USE_AMP):
            if Config.DEEP_SUPERVISION:
                # Get main output + auxiliary outputs
                out_main, out_aux1, out_aux2 = model(images)
                
                # Calculate loss for each output
                loss_main = criterion(out_main, labels)
                loss_aux1 = criterion(out_aux1, labels)
                loss_aux2 = criterion(out_aux2, labels)
                
                # Weighted combination
                loss = (Config.DEEP_SUPERVISION_WEIGHTS[0] * loss_main +
                       Config.DEEP_SUPERVISION_WEIGHTS[1] * loss_aux1 +
                       Config.DEEP_SUPERVISION_WEIGHTS[2] * loss_aux2)
                
                outputs = out_main  # Use main output for metrics
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
        
        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Calculate Dice per class
        with torch.no_grad():
            dice_scores = calculate_dice_per_class(outputs, labels, Config.NUM_CLASSES)
            for class_idx, score in enumerate(dice_scores):
                all_dice_scores[class_idx].append(score)
        
        running_loss += loss.item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'dice_avg': f'{np.mean([s[-1] for s in all_dice_scores if s]):.4f}'
        })
    
    epoch_loss = running_loss / len(loader)
    mean_dice_scores = [np.mean(scores) if scores else 0.0 for scores in all_dice_scores]
    
    return epoch_loss, mean_dice_scores


def validate(model, loader, criterion, device, epoch):
    """Validate the model"""
    model.eval()
    
    running_loss = 0.0
    all_dice_scores = [[] for _ in range(Config.NUM_CLASSES)]
    
    pbar = tqdm(loader, desc=f'Epoch {epoch} [Val]  ')
    with torch.no_grad():
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            with autocast(enabled=Config.USE_AMP):
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            # Calculate Dice per class
            dice_scores = calculate_dice_per_class(outputs, labels, Config.NUM_CLASSES)
            for class_idx, score in enumerate(dice_scores):
                all_dice_scores[class_idx].append(score)
            
            running_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'dice_avg': f'{np.mean([s[-1] for s in all_dice_scores if s]):.4f}'
            })
    
    epoch_loss = running_loss / len(loader)
    mean_dice_scores = [np.mean(scores) if scores else 0.0 for scores in all_dice_scores]
    
    return epoch_loss, mean_dice_scores


def plot_training_curves(train_losses, val_losses, train_dice, val_dice, save_path):
    """Plot and save training curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot losses
    ax1.plot(train_losses, label='Train Loss', marker='o')
    ax1.plot(val_losses, label='Val Loss', marker='s')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot mean Dice scores
    train_dice_mean = [np.mean(scores[1:]) for scores in train_dice]  # Exclude background
    val_dice_mean = [np.mean(scores[1:]) for scores in val_dice]
    
    ax2.plot(train_dice_mean, label='Train Dice', marker='o')
    ax2.plot(val_dice_mean, label='Val Dice', marker='s')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Mean Dice Score')
    ax2.set_title('Mean Dice Score (excluding background)')
    ax2.legend()
    ax2.grid(True)
    ax2.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def train():
    """Main training function"""
    # Set random seed
    torch.manual_seed(Config.SEED)
    np.random.seed(Config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(Config.SEED)
    
    # Print configuration
    Config.print_config()
    Config.create_dirs()
    
    # Check device
    device = torch.device(Config.DEVICE)
    print(f"\n🚀 Using device: {device}")
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB\n")
    
    # Create data loaders
    print("📊 Loading data...")
    train_loader, val_loader, test_loader = get_data_loaders()
    
    # Create model
    print("\n🏗️  Creating model...")
    model = get_model()
    model = model.to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {total_params:,}")
    print(f"   Model size: ~{total_params * 4 / 1024 / 1024:.2f} MB")
    
    # Loss function (MONAI's combined Dice + CE loss)
    criterion = DiceCELoss(
        include_background=True,
        to_onehot_y=True,
        softmax=True,
        squared_pred=True,
        reduction='mean'
    )
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )
    
    # Mixed precision scaler
    scaler = GradScaler(enabled=Config.USE_AMP)
    
    # TensorBoard writer
    writer = None
    if TENSORBOARD_AVAILABLE:
        writer = SummaryWriter(log_dir=Config.LOG_DIR)
        print(f"\n📈 TensorBoard logging to: {Config.LOG_DIR}")
        print(f"   Run: tensorboard --logdir={Config.LOG_DIR}")
    
    # Training tracking
    train_losses = []
    val_losses = []
    train_dice_scores = []
    val_dice_scores = []
    best_val_dice = 0.0
    patience_counter = 0
    
    class_names = ['Background', 'Body', 'Bone', 'Bladder', 'Rectum', 'Prostate']
    
    print("\n" + "="*70)
    print("🚀 STARTING TRAINING")
    print("="*70)
    
    start_time = time.time()
    
    try:
        for epoch in range(1, Config.NUM_EPOCHS + 1):
            epoch_start = time.time()
            
            # Train
            train_loss, train_dice = train_epoch(
                model, train_loader, criterion, optimizer, scaler, device, epoch
            )
            
            # Validate
            val_loss, val_dice = validate(
                model, val_loader, criterion, device, epoch
            )
            
            # Update scheduler
            scheduler.step(val_loss)
            
            # Store metrics
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_dice_scores.append(train_dice)
            val_dice_scores.append(val_dice)
            
            # Calculate mean Dice (excluding background)
            mean_val_dice = np.mean(val_dice[1:])
            
            # Log to TensorBoard
            if writer:
                writer.add_scalar('Loss/train', train_loss, epoch)
                writer.add_scalar('Loss/val', val_loss, epoch)
                writer.add_scalar('Dice/train_mean', np.mean(train_dice[1:]), epoch)
                writer.add_scalar('Dice/val_mean', mean_val_dice, epoch)
                
                for i, name in enumerate(class_names):
                    writer.add_scalar(f'Dice_class/train_{name}', train_dice[i], epoch)
                    writer.add_scalar(f'Dice_class/val_{name}', val_dice[i], epoch)
            
            # Print epoch summary
            epoch_time = time.time() - epoch_start
            print(f"\n{'='*70}")
            print(f"Epoch {epoch}/{Config.NUM_EPOCHS} - Time: {epoch_time:.1f}s")
            print(f"{'='*70}")
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"Mean Dice (train): {np.mean(train_dice[1:]):.4f} | Mean Dice (val): {mean_val_dice:.4f}")
            print(f"\nPer-class Dice scores (Validation):")
            for i, name in enumerate(class_names):
                print(f"  {name:12s}: {val_dice[i]:.4f}")
            
            # Save checkpoint
            if epoch % Config.SAVE_FREQ == 0:
                checkpoint_path = os.path.join(
                    Config.CHECKPOINT_DIR,
                    f'checkpoint_epoch_{epoch}.pth'
                )
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'val_dice': val_dice,
                }, checkpoint_path)
                print(f"\n💾 Checkpoint saved: {checkpoint_path}")
            
            # Save best model
            if mean_val_dice > best_val_dice:
                best_val_dice = mean_val_dice
                patience_counter = 0
                
                best_model_path = os.path.join(Config.CHECKPOINT_DIR, 'best_model.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_dice': val_dice,
                    'mean_val_dice': mean_val_dice,
                }, best_model_path)
                print(f"🏆 New best model saved! Mean Dice: {best_val_dice:.4f}")
            else:
                patience_counter += 1
                print(f"⏳ Patience: {patience_counter}/{Config.PATIENCE}")
            
            # Early stopping
            if patience_counter >= Config.PATIENCE:
                print(f"\n⏹️  Early stopping triggered after {epoch} epochs")
                break
            
            # Plot curves
            if epoch % 5 == 0:
                plot_path = os.path.join(Config.RESULTS_DIR, 'training_curves.png')
                plot_training_curves(
                    train_losses, val_losses,
                    train_dice_scores, val_dice_scores,
                    plot_path
                )
        
        total_time = time.time() - start_time
        print(f"\n{'='*70}")
        print(f"✅ TRAINING COMPLETE!")
        print(f"{'='*70}")
        print(f"Total time: {total_time/3600:.2f} hours")
        print(f"Best validation Dice: {best_val_dice:.4f}")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        
    finally:
        if writer:
            writer.close()
        
        # Final plot
        if len(train_losses) > 0:
            plot_path = os.path.join(Config.RESULTS_DIR, 'final_training_curves.png')
            plot_training_curves(
                train_losses, val_losses,
                train_dice_scores, val_dice_scores,
                plot_path
            )
            print(f"\n📊 Training curves saved to: {plot_path}")


if __name__ == "__main__":
    train()