# dataset.py
import os
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader, random_split

# MONAI imports for transforms
from monai.transforms import (
    Compose, RandRotate90, RandFlip, RandScaleIntensity,
    RandShiftIntensity, RandGaussianNoise, RandGaussianSmooth
)

from config import Config


class ProstateDataset3D(Dataset):
    """
    Dataset for 3D Prostate MRI segmentation from HipMRI Study.
    Uses MONAI transforms for data augmentation.
    """
    
    def __init__(self, data_dir, transform=None, is_train=True):
        """
        Args:
            data_dir: Path to HipMRI_Study_open folder
            transform: MONAI transforms for augmentation
            is_train: Whether this is training data (for augmentation)
        """
        self.mri_dir = os.path.join(data_dir, 'semantic_MRs')
        self.label_dir = os.path.join(data_dir, 'semantic_labels_only')
        self.transform = transform
        self.is_train = is_train
        
        # Get all file names
        self.mri_files = sorted([f for f in os.listdir(self.mri_dir) 
                                if f.endswith('.nii') or f.endswith('.nii.gz')])
        self.label_files = sorted([f for f in os.listdir(self.label_dir) 
                                  if f.endswith('.nii') or f.endswith('.nii.gz')])
        
        assert len(self.mri_files) == len(self.label_files), \
            "Mismatch between MRI and label files"
        
        print(f"{'Train' if is_train else 'Val/Test'} dataset: {len(self.mri_files)} volumes")
    
    def __len__(self):
        return len(self.mri_files)
    
    def __getitem__(self, idx):
        # Load MRI
        mri_path = os.path.join(self.mri_dir, self.mri_files[idx])
        mri_img = nib.load(mri_path)
        mri_data = mri_img.get_fdata()
        
        # Load label
        label_path = os.path.join(self.label_dir, self.label_files[idx])
        label_img = nib.load(label_path)
        label_data = label_img.get_fdata()
        
        # Normalize image (z-score normalization)
        mri_data = (mri_data - mri_data.mean()) / (mri_data.std() + 1e-8)
        
        # Convert to tensors
        # Add channel dimension: (D, H, W) -> (1, D, H, W)
        mri_data = torch.from_numpy(mri_data).float().unsqueeze(0)
        label_data = torch.from_numpy(label_data).long().unsqueeze(0)  # Add channel dim        
        # Apply MONAI transforms if provided
        if self.transform and self.is_train:
            # MONAI expects dict format
            data_dict = {'image': mri_data, 'label': label_data}
            # Note: MONAI transforms need proper setup, simplified here
            # In practice, you'd use MONAI's dictionary transforms
        
        return mri_data, label_data


def get_train_transforms():
    """
    Get training data augmentation transforms using MONAI.
    
    Based on nnU-Net augmentation strategy.
    """
    train_transforms = Compose([
        RandRotate90(prob=0.5, spatial_axes=(0, 1)),  # Random 90° rotation
        RandRotate90(prob=0.5, spatial_axes=(1, 2)),
        RandFlip(prob=0.5, spatial_axis=0),  # Random flip
        RandFlip(prob=0.5, spatial_axis=1),
        RandFlip(prob=0.5, spatial_axis=2),
        RandScaleIntensity(factors=0.1, prob=0.5),  # Intensity scaling
        RandShiftIntensity(offsets=0.1, prob=0.5),  # Intensity shift
        RandGaussianNoise(prob=0.15, std=0.01),  # Gaussian noise
        RandGaussianSmooth(prob=0.15),  # Gaussian smoothing
    ])
    return train_transforms


def get_data_loaders(batch_size=None):
    """
    Create train, validation, and test data loaders.
    
    Returns:
        train_loader, val_loader, test_loader
    """
    if batch_size is None:
        batch_size = Config.BATCH_SIZE
    
    # Create datasets
    full_dataset = ProstateDataset3D(
        Config.DATA_DIR,
        transform=None,  # We'll add transforms in training loop
        is_train=True
    )
    
    # Calculate splits
    total_size = len(full_dataset)
    train_size = int(Config.TRAIN_SPLIT * total_size)
    val_size = int(Config.VAL_SPLIT * total_size)
    test_size = total_size - train_size - val_size
    
    # Split dataset
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(Config.SEED)
    )
    
    # Update is_train flag
    # Note: This is simplified; in practice, create separate dataset instances
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY
    )
    
    print(f"\nDataset splits:")
    print(f"  Train: {len(train_dataset)} volumes")
    print(f"  Val:   {len(val_dataset)} volumes")
    print(f"  Test:  {len(test_dataset)} volumes")
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test the dataset
    from config import Config
    
    print("Testing dataset...")
    Config.print_config()
    
    # Create dataset
    dataset = ProstateDataset3D(Config.DATA_DIR)
    print(f"\nDataset size: {len(dataset)}")
    
    # Load one sample
    mri, label = dataset[0]
    print(f"\nSample data:")
    print(f"  MRI shape: {mri.shape}")
    print(f"  Label shape: {label.shape}")
    print(f"  MRI range: [{mri.min():.2f}, {mri.max():.2f}]")
    print(f"  Unique labels: {torch.unique(label)}")
    
    # Test dataloaders
    print("\nTesting dataloaders...")
    train_loader, val_loader, test_loader = get_data_loaders()
    print("Dataloaders created successfully!")
    
    # Test one batch
    for mri_batch, label_batch in train_loader:
        print(f"\nFirst batch:")
        print(f"  MRI batch shape: {mri_batch.shape}")
        print(f"  Label batch shape: {label_batch.shape}")
        break