"""
Configuration for 3D Prostate Segmentation with Improved U-Net
FULL TRAINING with 128x128x128 patches
"""

import os
import platform
import socket
import torch

class Config:
    """Configuration for 3D Prostate Segmentation with Improved U-Net"""
    
    # Auto-detect environment
    system = platform.system()
    hostname = socket.gethostname()
    
    # Data path
    if system == "Windows":
        DATA_DIR = r"H:\HipMRI_Study_open"  
    else:
        DATA_DIR = "/home/groups/comp3710/HipMRI_Study_open"
    
    # Model parameters
    IN_CHANNELS = 1
    NUM_CLASSES = 6
    BASE_FEATURES = 32
    
    # Improved U-Net specific
    DEEP_SUPERVISION = True
    USE_INSTANCE_NORM = True
    
    # Training parameters 
    BATCH_SIZE = 1
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 10  # ← FULL TRAINING (not 3!)
    WEIGHT_DECAY = 1e-5
    
    # Mixed precision
    USE_AMP = True
    
    # Data splits
    TRAIN_SPLIT = 0.7
    VAL_SPLIT = 0.15
    
    # Patch size - 128³ for optimal performance
    PATCH_SIZE = (128, 128, 128)
    
    # Output directories
    CHECKPOINT_DIR = "./checkpoints_128"  # ← Not "_test"
    RESULTS_DIR = "./results_128"         # ← Not "_test"
    LOG_DIR = "./logs_128"                # ← Not "_test"
    
    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Random seed
    SEED = 42
    
    # Data loading
    NUM_WORKERS = 0
    PIN_MEMORY = True if torch.cuda.is_available() else False
    
    # Checkpointing
    SAVE_FREQ = 5  # ← Save every 5 epochs (not 1)
    
    # Early stopping
    PATIENCE = 5  # ← Real early stopping
    
    # Loss weights for deep supervision
    DEEP_SUPERVISION_WEIGHTS = [1.0, 0.5, 0.25]
    
    @classmethod
    def create_dirs(cls):
        """Create output directories"""
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.RESULTS_DIR, exist_ok=True)
        os.makedirs(cls.LOG_DIR, exist_ok=True)
    
    @classmethod
    def print_config(cls):
        """Print current configuration"""
        print("="*70)
        print("Configuration - Improved 3D U-Net [128³ PRODUCTION]")
        print("="*70)
        print(f"System          : {cls.system}")
        print(f"Hostname        : {cls.hostname}")
        print(f"Data directory  : {cls.DATA_DIR}")
        print(f"Data exists     : {os.path.exists(cls.DATA_DIR)}")
        
        if os.path.exists(cls.DATA_DIR):
            print(f"  ✅ Data accessible!")
            try:
                mri_path = os.path.join(cls.DATA_DIR, 'semantic_MRs')
                n_files = len([f for f in os.listdir(mri_path) if f.endswith(('.nii', '.nii.gz'))])
                print(f"  📊 Found {n_files} MRI volumes")
            except:
                pass
        else:
            print(f"  ❌ Data NOT accessible!")
            
        print(f"Device          : {cls.DEVICE}")
        if cls.DEVICE == "cuda":
            print(f"GPU Name        : {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory      : {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"Mixed Precision : {cls.USE_AMP}")
        print(f"Batch size      : {cls.BATCH_SIZE}")
        print(f"Patch size      : {cls.PATCH_SIZE}")
        print(f"Learning rate   : {cls.LEARNING_RATE}")
        print(f"Epochs          : {cls.NUM_EPOCHS}")
        print(f"Deep Supervision: {cls.DEEP_SUPERVISION}")
        print("="*70)


if __name__ == "__main__":
    Config.print_config()
