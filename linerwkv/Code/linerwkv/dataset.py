import os
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import random

def load_bsq(file_path, shape, dtype=np.uint16):
    """Load BSQ file and reshape to (H,W,C)"""
    with open(file_path, 'rb') as f:
        data = np.fromfile(f, dtype=dtype)
    return data.reshape(shape).astype(np.float32)

class PaviaUDataset(Dataset):
    def __init__(self, config, mode="train"):
        super().__init__()
        assert mode in ["train", "val", "test"]
        
        # Configuration
        self.mode = mode
        self.patch_size = config.pos_size
        self.n_bands = 16 if mode == "train" else 103  # Dynamic band selection
        
        # Load and normalize data
        self.data = load_bsq(
            file_path=os.path.join(config.dataset_dir, "PaviaU.bsq"),
            shape=(610, 340, 103)  # PaviaU dimensions
        )
        
        # Normalization (matches original paper)
        self.min_value = 0
        self.max_value = np.max(self.data) # Adjust based on your data stats
        self.data = np.clip(self.data, self.min_value, self.max_value)
        self.data = (self.data - self.min_value) / (self.max_value - self.min_value)
        
        # Train/val/test split
        self.indices = self._generate_indices(
            test_ratio=0.1, 
            val_ratio=0.1,
            max_train=8000 if mode=="train" else None,
            max_val=800 if mode=="val" else None
        )

    def _generate_indices(self, test_ratio, val_ratio, max_train, max_val):
        """Generate spatial indices with stratified splitting"""
        H, W, _ = self.data.shape
        coords = [(i,j) for i in range(H) for j in range(W)]
        
        # Initial split
        train_val, test = train_test_split(coords, test_size=test_ratio, random_state=42)
        train, val = train_test_split(train_val, test_size=val_ratio/(1-test_ratio), random_state=42)
        
        # Subsample if needed
        indices = {
            "train": train[:max_train] if max_train else train,
            "val": val[:max_val] if max_val else val,
            "test": test
        }
        return indices[self.mode]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # Spatial patching
        i, j = self.indices[idx]
        p = self.patch_size // 2
        patch = self.data[
            max(i-p,0):min(i+p,self.data.shape[0]),
            max(j-p,0):min(j+p,self.data.shape[1]),
            :
        ]
        
        # Padding for edge cases
        pad_h = self.patch_size - patch.shape[0]
        pad_w = self.patch_size - patch.shape[1]
        if pad_h > 0 or pad_w > 0:
            patch = np.pad(patch, ((0,pad_h),(0,pad_w),(0,0)), mode='reflect')

        # Spectral subsampling (matching original implementation)
        if self.mode == "train":
            start_ch = random.randint(0, 103 - self.n_bands)
            channels = slice(start_ch, start_ch + self.n_bands)
        else:
            channels = slice(0, 103)  # Use all bands for val/test
            
        patch = patch[:, :, channels]  # (H,W,C)

        # Autoregressive formatting (matches original paper)
        input_img = torch.FloatTensor(patch[:-1]).permute(2,0,1)  # (C,H-1,W)
        target = torch.FloatTensor(patch[1:]).permute(2,0,1)       # (C,H-1,W)
        
        return input_img, target
