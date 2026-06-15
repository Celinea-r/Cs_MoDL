import os, sys
current_dir = os.path.dirname(os.path.abspath(__file__))
core_models_dir = os.path.dirname(current_dir)
root_dir = os.path.dirname(core_models_dir)
sys.path.append(core_models_dir)
sys.path.append(root_dir)

import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.MoDL_2D_multicoil import UnrolledModel2D

try:
    from pytorch_msssim import ssim
except ImportError:
    print("pytorch_msssim not found. Please install via: pip install pytorch-msssim")
    def ssim(img1, img2, data_range, size_average):
        return torch.tensor(1.0, requires_grad=True).to(img1.device)

class L1_2DSSIM_Loss(nn.Module):
    def __init__(self, alpha=1.0, beta=0.1):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.l1_loss = nn.L1Loss()
        
    def forward(self, recon, target):
        l1 = self.l1_loss(recon, target)
        recon_mag = torch.abs(recon)
        target_mag = torch.abs(target)
        drange = target_mag.max() - target_mag.min()
        if drange == 0: drange = 1.0
        # 2D SSIM
        ssim_val = ssim(recon_mag, target_mag, data_range=drange, size_average=True)
        return self.alpha * l1 + self.beta * (1 - ssim_val)

def main():
    parser = argparse.ArgumentParser(description="Training 2D Optimized MoDL")
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--modl_lamda', type=float, default=0.05)
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = UnrolledModel2D(num_grad_steps=5, num_cg_steps=10, modl_lamda=args.modl_lamda).to(device)

    # Initialize DataLoader
    data_dir = os.path.join(root_dir, 'harvard_mock_data') # Use mock data for testing
    if not os.path.exists(data_dir):
        print(f"Data directory {data_dir} not found. Please run mock_harvard_data.py first.")
        return
        
    from utils.dataset import HarvardKSpaceDataset
    dataset = HarvardKSpaceDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = L1_2DSSIM_Loss()

    print(f"Starting training on {device} for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        
        for batch_idx, (kspace, maps, mask, target) in enumerate(dataloader):
            kspace = kspace.to(device)
            maps = maps.to(device)
            mask = mask.to(device)
            target = target.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            recon = model(kspace, maps, mask)
            
            # Loss calculation
            loss = criterion(recon, target)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{args.epochs}] - Loss: {avg_loss:.4f}")

if __name__ == "__main__":
    main()
