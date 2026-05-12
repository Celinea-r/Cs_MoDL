import os, sys
folder1_path = '../'
sys.path.append(folder1_path)

import logging
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.MoDL_3D_multicoil import UnrolledModel3D

try:
    from pytorch_msssim import ssim
except ImportError:
    logging.warning("pytorch_msssim not found. Please install via: pip install pytorch-msssim")
    def ssim(img1, img2, data_range, size_average):
        return torch.tensor(1.0, requires_grad=True).to(img1.device)

class L1_3DSSIM_Loss(nn.Module):
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
        ssim_val = ssim(recon_mag, target_mag, data_range=drange, size_average=True)
        return self.alpha * l1 + self.beta * (1 - ssim_val)

def main():
    parser = argparse.ArgumentParser(description="Training 3D MoDL for Harvard Precision Morphometry")
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--modl_lamda', type=float, default=0.05)
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = UnrolledModel3D(num_grad_steps=5, num_cg_steps=10, modl_lamda=args.modl_lamda).to(device)

if __name__ == "__main__":
    main()
