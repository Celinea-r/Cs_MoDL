import os
import torch
import numpy as np
from torch.utils.data import Dataset

class HarvardKSpaceDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        # Find all unique sample IDs by looking at the kspace files
        self.sample_ids = []
        for f in os.listdir(data_dir):
            if f.endswith('_kspace.npy'):
                self.sample_ids.append(f.replace('_kspace.npy', ''))
        self.sample_ids.sort()

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]
        
        # Load the arrays directly from the hard drive
        kspace = np.load(os.path.join(self.data_dir, f"{sample_id}_kspace.npy"))
        maps = np.load(os.path.join(self.data_dir, f"{sample_id}_maps.npy"))
        mask = np.load(os.path.join(self.data_dir, f"{sample_id}_mask.npy"))
        target = np.load(os.path.join(self.data_dir, f"{sample_id}_target.npy"))
        
        # Convert to PyTorch tensors
        kspace_t = torch.from_numpy(kspace)
        maps_t = torch.from_numpy(maps)
        mask_t = torch.from_numpy(mask)
        target_t = torch.from_numpy(target)
        
        return kspace_t, maps_t, mask_t, target_t
