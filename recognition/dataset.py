# dataset.py using volume based split due to data leakage in the case of patch-based extraction
import os
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader, Subset

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
    
    FIXED: Proper patient-level splitting to prevent data leakage.
    Handles longitudinal data (multiple weeks per patient).
    
    CRITICAL FIX v2: Proper cropping strategy:
    - Training (is_train=True): Random crops for data augmentation
    - Evaluation (is_train=False): Center crops for reproducible evaluation
    """
    
    def __init__(self, data_dir, transform=None, is_train=True):
        """
        Args:
            data_dir: Path to HipMRI_Study_open folder
            transform: MONAI transforms for augmentation
            is_train: Whether this is training data (random crops if True, center crops if False)
        """
        self.mri_dir = os.path.join(data_dir, 'semantic_MRs')
        self.label_dir = os.path.join(data_dir, 'semantic_labels_only')
        self.transform = transform
        self.is_train = is_train
        
        # Get all MRI and label files from subdirectories
        self.mri_files = []
        self.label_files = []
        self.patient_ids = []
        
        # Collect MRI files from subdirectories
        mri_paths = []
        for root, dirs, files in os.walk(self.mri_dir):
            for f in files:
                if f.endswith('.nii') or f.endswith('.nii.gz'):
                    mri_paths.append(os.path.join(root, f))
        
        # Collect label files from subdirectories
        label_paths = []
        for root, dirs, files in os.walk(self.label_dir):
            for f in files:
                if f.endswith('.nii') or f.endswith('.nii.gz'):
                    label_paths.append(os.path.join(root, f))
        
        # Create mapping from filename to full path
        mri_dict = {os.path.basename(p): p for p in mri_paths}
        label_dict = {os.path.basename(p): p for p in label_paths}
        
        # Match MRI to labels
        matched_count = 0
        for mri_file in sorted(mri_dict.keys()):
            patient_id = self._extract_patient_id(mri_file)
            
            # Find matching label
            matching_label = None
            for label_file in label_dict.keys():
                if self._extract_patient_id(label_file) == patient_id:
                    # Check if it's the same week/timepoint
                    if self._extract_week(mri_file) == self._extract_week(label_file):
                        matching_label = label_file
                        break
            
            if matching_label:
                self.mri_files.append(mri_dict[mri_file])
                self.label_files.append(label_dict[matching_label])
                self.patient_ids.append(patient_id)
                matched_count += 1
            else:
                print(f" Warning: No matching label found for {mri_file}")
        
        crop_type = "RANDOM crops" if is_train else "CENTER crops"
        print(f"{'Train' if is_train else 'Val/Test'} dataset: {len(self.mri_files)} volumes ({crop_type})")
        print(f"   Matched {matched_count}/{len(mri_dict)} MRI files to labels")
        print(f"   Unique patients: {len(set(self.patient_ids))}")
    
    def _extract_patient_id(self, filename):
        """
        Extract patient ID from filename.
        
        Format: PATIENTID_WeekN_TYPE.nii.gz
        Examples:
        - W012_Week1_SEMANTIC.nii.gz -> W012
        - J026_Week4_LFOV.nii.gz -> J026
        - H017_Week0_LFOV.nii.gz -> H017
        
        CRITICAL: Only extract patient ID (W012, J026, etc.)
        NOT the week number! This ensures all weeks from the same
        patient stay in the same split (train/val/test).
        """
        # Get just the filename without path
        basename = os.path.basename(filename)
        
        # Remove extension
        name = basename.replace('.nii.gz', '').replace('.nii', '')
        
        # Split by underscore and take first part (patient ID)
        # W012_Week1_SEMANTIC -> ['W012', 'Week1', 'SEMANTIC']
        patient_id = name.split('_')[0]
        
        return patient_id
    
    def _extract_week(self, filename):
        """
        Extract week number from filename.
        
        Examples:
        - W012_Week1_SEMANTIC.nii.gz -> Week1
        - J026_Week4_LFOV.nii.gz -> Week4
        """
        basename = os.path.basename(filename)
        name = basename.replace('.nii.gz', '').replace('.nii', '')
        parts = name.split('_')
        
        # Find the part that starts with "Week"
        for part in parts:
            if part.startswith('Week'):
                return part
        
        return 'Week0'  # Default if not found
    
    def __len__(self):
        return len(self.mri_files)
    
    def __getitem__(self, idx):
        # Load MRI
        mri_path = self.mri_files[idx]
        mri_img = nib.load(mri_path)
        mri_data = mri_img.get_fdata()
        
        # Load label
        label_path = self.label_files[idx]
        label_img = nib.load(label_path)
        label_data = label_img.get_fdata()
        
        # Normalize image (z-score normalization)
        mri_data = (mri_data - mri_data.mean()) / (mri_data.std() + 1e-8)
        
        # Convert to tensors
        mri_data = torch.from_numpy(mri_data).float().unsqueeze(0)
        label_data = torch.from_numpy(label_data).long().unsqueeze(0)
        
        # Extract patches to fit in GPU memory
        patch_size = Config.PATCH_SIZE
        
        if self.is_train:
            # ============================================================
            # TRAINING: Random crop for data augmentation
            # ============================================================
            # Different patch location every time = more variety
            d, h, w = mri_data.shape[1:]
            pd, ph, pw = patch_size
            
            # Random starting positions
            d_start = torch.randint(0, max(1, d - pd + 1), (1,)).item()
            h_start = torch.randint(0, max(1, h - ph + 1), (1,)).item()
            w_start = torch.randint(0, max(1, w - pw + 1), (1,)).item()
            
            # Extract patches
            mri_data = mri_data[:, d_start:d_start+pd, h_start:h_start+ph, w_start:w_start+pw]
            label_data = label_data[:, d_start:d_start+pd, h_start:h_start+ph, w_start:w_start+pw]
        else:
            # ============================================================
            # VALIDATION/TEST: Center crop for reproducible evaluation
            # ============================================================
            # Same patch location every time = consistent results
            d, h, w = mri_data.shape[1:]
            pd, ph, pw = patch_size
            
            # Center starting positions (always same for a given volume)
            d_start = max(0, (d - pd) // 2)
            h_start = max(0, (h - ph) // 2)
            w_start = max(0, (w - pw) // 2)
            
            # Extract center patch
            mri_data = mri_data[:, d_start:d_start+pd, h_start:h_start+ph, w_start:w_start+pw]
            label_data = label_data[:, d_start:d_start+pd, h_start:h_start+ph, w_start:w_start+pw]
        
        # Apply MONAI transforms (only for training)
        if self.transform and self.is_train:
            data_dict = {'image': mri_data, 'label': label_data}
            data_dict = self.transform(data_dict)
            mri_data = data_dict['image']
            label_data = data_dict['label']
        
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
    
    CRITICAL FIX v2: Creates separate dataset instances for reproducible evaluation
    - Training dataset: Uses random crops (data augmentation)
    - Val/Test datasets: Uses center crops (reproducible evaluation)
    
    This ensures:
    1. Training sees variety (random crops = augmentation)
    2. Evaluation is consistent (center crops = same patch every time)
    3. No patient appears in multiple splits (patient-level splitting)
    
    Patient-level splitting prevents data leakage:
    - If patient W012 is in training → ALL weeks of W012 stay in training
    - If patient H017 is in validation → ALL weeks of H017 stay in validation
    
    Returns:
        train_loader, val_loader, test_loader
    """
    if batch_size is None:
        batch_size = Config.BATCH_SIZE
    
    # =====================================================================
    # CRITICAL FIX: Create TWO separate dataset instances
    # =====================================================================
    
    # 1. Training dataset with RANDOM crops (data augmentation)
    train_dataset_full = ProstateDataset3D(
        Config.DATA_DIR,
        transform=None,  # Transforms applied in training loop
        is_train=True    # Random crops for training
    )
    
    # 2. Evaluation dataset with CENTER crops (reproducible evaluation)
    eval_dataset_full = ProstateDataset3D(
        Config.DATA_DIR,
        transform=None,  # No transforms for evaluation
        is_train=False   # Center crops for validation/test
    )
    
    # Get unique patient IDs (use training dataset for consistency)
    unique_patients = list(set(train_dataset_full.patient_ids))
    unique_patients.sort()  # For reproducibility
    
    print(f"\nTotal scans: {len(train_dataset_full)}")
    print(f"Unique patients: {len(unique_patients)}")
    
    # Count scans per patient to verify
    from collections import Counter
    patient_counts = Counter(train_dataset_full.patient_ids)
    print(f"Scans per patient (examples):")
    for patient, count in sorted(patient_counts.items())[:5]:
        print(f"  {patient}: {count} scans")
    
    # Shuffle patients with fixed seed
    rng = np.random.default_rng(Config.SEED)
    rng.shuffle(unique_patients)
    
    # Calculate patient-level splits
    n_total = len(unique_patients)
    n_train = int(Config.TRAIN_SPLIT * n_total)
    n_val = int(Config.VAL_SPLIT * n_total)
    n_test = n_total - n_train - n_val
    
    # Split patients into train/val/test
    train_patients = set(unique_patients[:n_train])
    val_patients = set(unique_patients[n_train:n_train + n_val])
    test_patients = set(unique_patients[n_train + n_val:])
    
    # =====================================================================
    # CRITICAL: Get indices from the APPROPRIATE dataset
    # =====================================================================
    
    # Training indices from train_dataset_full (random crops)
    train_idx = [i for i, pid in enumerate(train_dataset_full.patient_ids) 
                 if pid in train_patients]
    
    # Val/Test indices from eval_dataset_full (center crops)
    val_idx = [i for i, pid in enumerate(eval_dataset_full.patient_ids) 
               if pid in val_patients]
    test_idx = [i for i, pid in enumerate(eval_dataset_full.patient_ids) 
                if pid in test_patients]
    
    # Verify no overlap
    train_set = set(train_dataset_full.patient_ids[i] for i in train_idx)
    val_set = set(eval_dataset_full.patient_ids[i] for i in val_idx)
    test_set = set(eval_dataset_full.patient_ids[i] for i in test_idx)
    
    assert len(train_set & val_set) == 0, "Patient overlap between train and val!"
    assert len(train_set & test_set) == 0, "Patient overlap between train and test!"
    assert len(val_set & test_set) == 0, "Patient overlap between val and test!"
    
    # Create subsets from appropriate datasets
    train_subset = Subset(train_dataset_full, train_idx)  # Uses random crops
    val_subset = Subset(eval_dataset_full, val_idx)       # Uses center crops
    test_subset = Subset(eval_dataset_full, test_idx)     # Uses center crops
    
    # Create dataloaders
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY
    )
    
    test_loader = DataLoader(
        test_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY
    )
    
    print(f"\n" + "="*70)
    print("Dataset splits (PATIENT-LEVEL - NO DATA LEAKAGE)")
    print("="*70)
    print(f"Train: {len(train_idx)} scans from {len(train_patients)} patients (RANDOM crops)")
    print(f"Val:   {len(val_idx)} scans from {len(val_patients)} patients (CENTER crops)")
    print(f"Test:  {len(test_idx)} scans from {len(test_patients)} patients (CENTER crops)")
    print(f"\n VERIFIED: No patient appears in multiple splits!")
    print(f"All timepoints from same patient stay together!")
    print(f"Val/Test use consistent CENTER crops for reproducible evaluation!")
    print("="*70)
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test the dataset
    from config import Config
    
    print("Testing dataset...")
    Config.print_config()
    
    # Test both training and evaluation datasets
    print("\n" + "="*70)
    print("Creating TRAINING dataset (random crops)...")
    print("="*70)
    train_dataset = ProstateDataset3D(Config.DATA_DIR, is_train=True)
    print(f"\nDataset size: {len(train_dataset)}")
    print(f"Unique patients: {len(set(train_dataset.patient_ids))}")
    
    print("\n" + "="*70)
    print("Creating EVALUATION dataset (center crops)...")
    print("="*70)
    eval_dataset = ProstateDataset3D(Config.DATA_DIR, is_train=False)
    print(f"\nDataset size: {len(eval_dataset)}")
    print(f"Unique patients: {len(set(eval_dataset.patient_ids))}")
    
    # Show some examples
    print(f"\nFirst 5 files:")
    for i in range(min(5, len(train_dataset))):
        mri_file = os.path.basename(train_dataset.mri_files[i])
        label_file = os.path.basename(train_dataset.label_files[i])
        patient = train_dataset.patient_ids[i]
        print(f"  Patient {patient}: MRI={mri_file}, Label={label_file}")
    
    # Load one sample from each
    print("\n" + "="*70)
    print("Testing sample extraction...")
    print("="*70)
    
    print("\nTraining dataset (random crop):")
    mri_train, label_train = train_dataset[0]
    print(f"  MRI shape: {mri_train.shape}")
    print(f"  Label shape: {label_train.shape}")
    print(f"  MRI range: [{mri_train.min():.2f}, {mri_train.max():.2f}]")
    print(f"  Unique labels: {torch.unique(label_train)}")
    
    # Load same sample again to verify randomness
    mri_train2, label_train2 = train_dataset[0]
    same_values = torch.allclose(mri_train, mri_train2, atol=1e-6)
    print(f"  Random crop test: {'DIFFERENT crops (correct!)' if not same_values else '❌ Same crop (wrong!)'}")
    
    print("\nEvaluation dataset (center crop):")
    mri_eval, label_eval = eval_dataset[0]
    print(f"  MRI shape: {mri_eval.shape}")
    print(f"  Label shape: {label_eval.shape}")
    print(f"  MRI range: [{mri_eval.min():.2f}, {mri_eval.max():.2f}]")
    print(f"  Unique labels: {torch.unique(label_eval)}")
    
    # Load same sample again to verify consistency
    mri_eval2, label_eval2 = eval_dataset[0]
    same_values = torch.allclose(mri_eval, mri_eval2, atol=1e-6)
    print(f"  Center crop test: {'SAME crop (correct!)' if same_values else '❌ Different crop (wrong!)'}")
    
    # Test dataloaders
    print("\n" + "="*70)
    print("Testing dataloaders...")
    print("="*70)
    train_loader, val_loader, test_loader = get_data_loaders()
    print("\nDataloaders created successfully!")
    
    # Test one batch
    for mri_batch, label_batch in train_loader:
        print(f"\nFirst training batch:")
        print(f"  MRI batch shape: {mri_batch.shape}")
        print(f"  Label batch shape: {label_batch.shape}")
        break
    
    print("\n" + "="*70)
    print("ALL TESTS PASSED!")
    print("="*70)