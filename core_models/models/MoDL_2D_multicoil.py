import torch
import torch.nn as nn
import sys

# Import the 2D UNet
from models.unet.unet_model import UNet

class SenseModel_2D(nn.Module):
    """
    A module that computes 2D forward and adjoint SENSE operations.
    x: [B, 1, H, W] (complex)
    maps: [B, C, H, W] (complex)
    mask: [B, 1, H, W] (float or complex)
    """
    def __init__(self, maps, mask, l2lam=0.0):
        super().__init__()
        self.maps = maps
        self.mask = mask
        self.l2lam = l2lam

    def forward_op(self, image):
        # A(x) = M * F(S * x)
        kspace = image * self.maps
        # 2D FFT
        kspace = torch.fft.fftn(kspace, dim=(-2, -1), norm="ortho")
        kspace = kspace * self.mask
        return kspace

    def adjoint_op(self, kspace):
        # A^H(y) = S^H * F^-1(M * y)
        masked_kspace = kspace * self.mask
        # 2D IFFT
        image_coils = torch.fft.ifftn(masked_kspace, dim=(-2, -1), norm="ortho")
        # Coil combination
        image = torch.sum(image_coils * self.maps.conj(), dim=1, keepdim=True)
        return image

    def normal(self, x):
        # A^H A x + lambda x
        out = self.adjoint_op(self.forward_op(x))
        if self.l2lam > 0:
            out = out + self.l2lam * x
        return out

def dot_batch_2d(x1, x2):
    return torch.sum(x1.conj() * x2, dim=list(range(1, len(x1.shape)))).real.view(-1, *(1,)*(len(x1.shape)-1))

def ip_batch_2d(x):
    return dot_batch_2d(x, x)

def conjgrad_2d(x, b, Aop_fun, max_iter=10, eps=1e-4):
    """
    Differentiable Conjugate Gradient solver for complex 2D tensors.
    """
    r = b - Aop_fun(x)
    p = r.clone()

    rsnot = ip_batch_2d(r)
    rsold = rsnot.clone()
    rsnew = rsnot.clone()

    for i in range(max_iter):
        if rsnew.max() < eps:
            break

        Ap = Aop_fun(p)
        pAp = dot_batch_2d(p, Ap)

        alpha = rsold / pAp
        x = x + alpha * p
        r = r - alpha * Ap

        rsnew = ip_batch_2d(r)
        beta = rsnew / rsold
        rsold = rsnew.clone()

        p = beta * p + r

    return x

class UnrolledModel2D(nn.Module):
    """
    2D Unrolled Compressed Sensing.
    """
    def __init__(self, num_grad_steps=5, num_cg_steps=10, modl_lamda=0.05):
        super().__init__()
        
        self.num_grad_steps = num_grad_steps
        self.num_cg_steps = num_cg_steps
        self.modl_lamda = modl_lamda

        self.unets = nn.ModuleList([
            UNet(n_channels=2, n_classes=2)
            for _ in range(self.num_grad_steps)
        ])

    def forward(self, kspace, maps, mask):
        A = SenseModel_2D(maps, mask, l2lam=self.modl_lamda)
        image = A.adjoint_op(kspace)
        
        for unet in self.unets:
            # Prepare for 2D UNet (Real/Imaginary as channels)
            image_real = image.real
            image_imag = image.imag
            image_2ch = torch.cat([image_real, image_imag], dim=1) # [B, 2, H, W]
            
            image_2ch = unet(image_2ch)
            image_reg = torch.complex(image_2ch[:, 0:1, ...], image_2ch[:, 1:2, ...])
            
            # Data Consistency using CG
            rhs = A.adjoint_op(kspace) + self.modl_lamda * image_reg
            image = conjgrad_2d(x=image, b=rhs, Aop_fun=A.normal, max_iter=self.num_cg_steps)
            
        return image
