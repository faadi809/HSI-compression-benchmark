import numpy as np
import os
import torch
from scipy.io import loadmat
from torch.utils.data import Dataset
from einops import rearrange
import random
import csv

class PaviaUDataset(Dataset):
    """
    Pavia University Hyperspectral Dataset
    Expected file structure:
    - root_dir/
        - PaviaU.mat (containing 'paviaU' array)
        - splits/ (will be created automatically)
            - train.csv
            - val.csv
            - test.csv
    
    Splits will be created with 70% train, 15% val, 15% test ratio if not existing
    """
    def __init__(self, root_dir, split="train", transform=None, random_subsample_factor=None, 
                 patch_size=64, test_size=0.15, val_size=0.15, random_state=42):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.random_subsample_factor = random_subsample_factor
        self.patch_size = patch_size
        
        # Load and preprocess data
        mat_data = loadmat(os.path.join(root_dir, '/home/data/Fahad/HSI datasets/PaviaU/PaviaU.mat'))
        self.data = mat_data['paviaU']  # Key might need adjustment
        
        # Convert to CxHxW and normalize
        if len(self.data.shape) == 3:
            self.data = np.transpose(self.data, (2, 0, 1))
        self.data = self.data.astype(np.float32)
        self.data = (self.data - self.data.min()) / (self.data.max() - self.data.min())
        
        # Create or load patch indices
        self.patch_coords = self._create_or_load_patch_splits(test_size, val_size, random_state)
        
    def _create_or_load_patch_splits(self, test_size, val_size, random_state):
        split_dir = os.path.join(self.root_dir, 'splits')
        os.makedirs(split_dir, exist_ok=True)
        
        # Generate all possible patch coordinates
        H, W = self.data.shape[1], self.data.shape[2]
        coords = []
        for y in range(0, H - self.patch_size + 1, self.patch_size):
            for x in range(0, W - self.patch_size + 1, self.patch_size):
                coords.append((y, x))
        
        # Create splits if they don't exist
        if not all(os.path.exists(os.path.join(split_dir, f)) for f in ['train.csv', 'val.csv', 'test.csv']):
            random.seed(random_state)
            random.shuffle(coords)
            
            n = len(coords)
            n_test = int(n * test_size)
            n_val = int(n * val_size)
            
            splits = {
                'train': coords[n_test+n_val:],
                'val': coords[n_test:n_test+n_val],
                'test': coords[:n_test]
            }
            
            # Save splits as CSV files
            for split_name, split_coords in splits.items():
                with open(os.path.join(split_dir, f'{split_name}.csv'), 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerows([list(coord) for coord in split_coords])
        else:
            # Load existing splits
            splits = {}
            for split_name in ['train', 'val', 'test']:
                with open(os.path.join(split_dir, f'{split_name}.csv'), 'r') as f:
                    reader = csv.reader(f)
                    splits[split_name] = [tuple(map(int, row)) for row in reader]
        
        return splits[self.split]
    
    def __len__(self):
        return len(self.patch_coords)
    
    def __getitem__(self, index):
        y, x = self.patch_coords[index]
        img = self.data[:, y:y+self.patch_size, x:x+self.patch_size]
        
        # Apply random subsampling if specified
        if self.random_subsample_factor:
            c, h, w = img.shape
            sample_size = int(h / self.random_subsample_factor) ** 2
            flattened_tensor = torch.from_numpy(img).flatten(1, 2)
            random_indices = torch.randperm(flattened_tensor.size(1))[:sample_size]
            subsampled_tensor = flattened_tensor[:, random_indices]
            img = rearrange(subsampled_tensor, 'c (h w) -> c h w',
                          h=int(h/self.random_subsample_factor),
                          w=int(w/self.random_subsample_factor))
        else:
            img = torch.from_numpy(img)
        
        if self.transform:
            img = self.transform(img)
            
        return img
