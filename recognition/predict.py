# predict.py
"""
Demonstration script for 3D prostate segmentation inference.
Loads trained model and runs prediction on test samples with visualization.
"""

import torch
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import os

from modules import get_model
from dataset import get_data_loaders
from config import Config

# ======================
# MODEL LOADING FUNCTION
# ======================
def load_trained_model(checkpoint_path="checkpoints/best_model.pth", device=None):
    """
    Load a trained 3D U-Net model safely with PyTorch 2.6+.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("🏗️  Loading trained model...")

    # Create model instance
    model = get_model()
    model.to(device)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    # Add safe globals for numpy types
    torch.serialization.add_safe_globals([np._core.multiarray.scalar, np.dtype])

    # Load checkpoint weights safely
    #torch.serialization.add_safe_globals([
    #np._core.multiarray.scalar,
    #np.dtype,
    #np.float64
    #])
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    #checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Extract checkpoint info
    checkpoint_info = {
        'epoch': checkpoint.get('epoch', 'unknown'),
        'val_dice': checkpoint.get('mean_val_dice', checkpoint.get('best_dice', 'unknown'))
    }

    print(f"✅ Loaded model from epoch {checkpoint_info['epoch']}")
    if isinstance(checkpoint_info['val_dice'], float):
        print(f"   Validation Dice: {checkpoint_info['val_dice']:.4f}")

    return model, checkpoint_info

# ======================
# PREDICTION FUNCTION
# ======================
def predict_single_volume(model, mri, device):
    model.eval()
    with torch.no_grad():
        if mri.dim() == 4:
            mri = mri.unsqueeze(0)
        mri = mri.to(device)
        output = model(mri)
        if isinstance(output, tuple):
            output = output[0]  # Handle deep supervision
        prediction = torch.argmax(output, dim=1).squeeze().cpu().numpy()
    return prediction

# ======================
# DICE CALCULATION
# ======================
def calculate_dice(pred, target, class_id):
    pred_mask = (pred == class_id)
    target_mask = (target == class_id)
    intersection = np.sum(pred_mask & target_mask)
    union = np.sum(pred_mask) + np.sum(target_mask)
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return (2.0 * intersection) / union

# ======================
# VISUALIZATION FUNCTIONS
# ======================
def visualize_prediction(mri, label, prediction, slice_idx=None, save_path='prediction_visualization.png'):
    if mri.ndim == 4: mri = mri[0]
    if label.ndim == 4: label = label[0]
    if slice_idx is None: slice_idx = mri.shape[0] // 2

    mri_slice = mri[slice_idx, :, :]
    label_slice = label[slice_idx, :, :]
    pred_slice = prediction[slice_idx, :, :]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    class_names = ['Background', 'Body', 'Bone', 'Bladder', 'Rectum', 'Prostate']

    axes[0,0].imshow(mri_slice, cmap='gray'); axes[0,0].set_title('MRI'); axes[0,0].axis('off')
    axes[0,1].imshow(label_slice, cmap='tab10', vmin=0, vmax=5); axes[0,1].set_title('Ground Truth'); axes[0,1].axis('off')
    axes[0,2].imshow(pred_slice, cmap='tab10', vmin=0, vmax=5); axes[0,2].set_title('Prediction'); axes[0,2].axis('off')

    sagittal_idx = mri.shape[2] // 2
    mri_sag = mri[:, :, sagittal_idx]
    label_sag = label[:, :, sagittal_idx]
    pred_sag = prediction[:, :, sagittal_idx]

    axes[1,0].imshow(mri_sag, cmap='gray'); axes[1,0].set_title('MRI'); axes[1,0].axis('off')
    axes[1,1].imshow(label_sag, cmap='tab10', vmin=0, vmax=5); axes[1,1].set_title('Ground Truth'); axes[1,1].axis('off')
    axes[1,2].imshow(pred_sag, cmap='tab10', vmin=0, vmax=5); axes[1,2].set_title('Prediction'); axes[1,2].axis('off')

    cbar = plt.colorbar(axes[0,2].images[0], ax=axes, orientation='horizontal', pad=0.05, fraction=0.05)
    cbar.set_ticks(range(6)); cbar.set_ticklabels(class_names); cbar.ax.tick_params(labelsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 Visualization saved to: {save_path}")

def visualize_prostate_only(mri, label, prediction, save_path='prostate_overlay.png'):
    """
    Visualize only the prostate overlay on the MRI slice with the most prostate voxels.
    """
    if mri.ndim == 4: mri = mri[0]
    if label.ndim == 4: label = label[0]

    prostate_counts = np.sum(label == 5, axis=(1,2))
    if prostate_counts.max() == 0: return
    best_slice = np.argmax(prostate_counts)

    mri_slice = mri[best_slice]
    label_slice = (label[best_slice] == 5).astype(float)
    pred_slice = (prediction[best_slice] == 5).astype(float)

    fig, axes = plt.subplots(1, 3, figsize=(15,5))
    axes[0].imshow(mri_slice, cmap='gray'); axes[0].set_title('MRI'); axes[0].axis('off')
    axes[1].imshow(mri_slice, cmap='gray'); axes[1].imshow(label_slice, cmap='Reds', alpha=0.5); axes[1].set_title('GT Prostate'); axes[1].axis('off')
    axes[2].imshow(mri_slice, cmap='gray'); axes[2].imshow(pred_slice, cmap='Blues', alpha=0.5); axes[2].set_title('Pred Prostate'); axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"📊 Prostate overlay saved to: {save_path}")

# ======================
# MAIN FUNCTION
# ======================
def main():
    print("\n" + "="*70)
    print("3D PROSTATE SEGMENTATION - PREDICTION DEMONSTRATION")
    print("="*70)
    Config.print_config()
    device = torch.device(Config.DEVICE)
    os.makedirs(Config.RESULTS_DIR, exist_ok=True)

    # Load model
    try:
        checkpoint_path = os.path.join("checkpoints", "best_model.pth")
        model, checkpoint_info = load_trained_model(checkpoint_path=checkpoint_path, device=device)
    except FileNotFoundError:
        print("❌ No trained model found! Train first using python train.py")
        return

    # Load test data
    _, _, test_loader = get_data_loaders()
    print(f"Test set size: {len(test_loader.dataset)} samples")

    num_samples = min(3, len(test_loader.dataset))
    class_names = ['Background', 'Body', 'Bone', 'Bladder', 'Rectum', 'Prostate']

    for i in range(num_samples):
        print(f"\n--- Sample {i+1}/{num_samples} ---")
        mri, label = test_loader.dataset[i]
        prediction = predict_single_volume(model, mri, device)

        # Dice scores
        if label.ndim == 4: label_np = label[0].numpy()
        else: label_np = label.numpy()
        print("Dice scores per class:")
        for class_id, name in enumerate(class_names):
            dice = calculate_dice(prediction, label_np, class_id)
            if np.sum(label_np == class_id) > 0:
                print(f"  {name:12s}: {dice:.4f}")

        # Visualizations
        if label.ndim == 4: label_vis = label[0].numpy()
        else: label_vis = label.numpy()
        if mri.ndim == 4: mri_vis = mri[0].numpy()
        else: mri_vis = mri.numpy()

        save_path_full = os.path.join(Config.RESULTS_DIR, f'prediction_sample_{i+1}.png')
        visualize_prediction(mri_vis, label_vis, prediction, save_path=save_path_full)

        save_path_prostate = os.path.join(Config.RESULTS_DIR, f'prostate_sample_{i+1}.png')
        visualize_prostate_only(mri_vis, label_vis, prediction, save_path=save_path_prostate)

    print("\nPrediction demonstration complete!")

if __name__ == "__main__":
    main()
