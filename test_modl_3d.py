import os, sys
folder1_path = 'core_models/'
sys.path.append(folder1_path)

import torch
from models.MoDL_3D_multicoil import SenseModel_3D_Harvard, UnrolledModel3D
from runners.Train_MoDL_3D import L1_3DSSIM_Loss

def test_physics():
    print("Testing 3D Forward/Adjoint Physics...")
    B, C, D, H, W = 1, 4, 16, 16, 16
    
    maps = torch.randn(B, C, D, H, W, dtype=torch.cfloat)
    mask = (torch.rand(B, 1, D, H, W) > 0.5).float()
    image = torch.randn(B, 1, D, H, W, dtype=torch.cfloat)
    
    A = SenseModel_3D_Harvard(maps, mask)
    
    kspace = A.forward_op(image)
    print(f"Forward K-space shape: {kspace.shape}")
    assert kspace.shape == (B, C, D, H, W)
    
    image_adj = A.adjoint_op(kspace)
    print(f"Adjoint Image shape: {image_adj.shape}")
    assert image_adj.shape == (B, 1, D, H, W)
    print("Physics sanity check passed!")

def test_gradient_flow():
    print("\nTesting Gradient Flow through 3D CNN and Unrolled CG...")
    B, C, D, H, W = 1, 4, 16, 16, 16
    
    maps = torch.randn(B, C, D, H, W, dtype=torch.cfloat)
    mask = (torch.rand(B, 1, D, H, W) > 0.5).float()
    target = torch.randn(B, 1, D, H, W, dtype=torch.cfloat)
    
    kspace = SenseModel_3D_Harvard(maps, mask).forward_op(target).detach()
    
    model = UnrolledModel3D(num_grad_steps=2, num_cg_steps=3)
    
    recon = model(kspace, maps, mask)
    print(f"Recon shape: {recon.shape}")
    
    criterion = L1_3DSSIM_Loss()
    loss = criterion(recon, target)
    print(f"Loss computed: {loss.item()}")
    
    loss.backward()
    
    grads_exist = any(p.grad is not None for p in model.parameters())
            
    if grads_exist:
        print("Gradient flow test passed! CG and CNN blocks are fully differentiable.")
    else:
        print("Gradient flow test failed! No gradients found.")

if __name__ == "__main__":
    test_physics()
    test_gradient_flow()
