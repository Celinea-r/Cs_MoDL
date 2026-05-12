import torch
import torch.nn as nn
import sys
from utils.layers3D import ResNet

class SenseModel_3D_Harvard(nn.Module):
    """
    A module that computes 3D forward and adjoint SENSE operations for the Harvard Protocol.
    x: [B, 1, D, H, W] (complex)
    maps: [B, C, D, H, W] (complex)
    mask: [B, 1, D, H, W] (float or complex)
    """
    def __init__(self, maps, mask, l2lam=0.0):
        super().__init__()
        self.maps = maps
        self.mask = mask
        self.l2lam = l2lam

    def forward_op(self, image):
        # A(x) = M * F(S * x)
        # S * x
        kspace = image * self.maps
        # F(...)
        kspace = torch.fft.fftn(kspace, dim=(-3, -2, -1), norm="ortho")
        # M * ...
        kspace = kspace * self.mask
        return kspace

    def adjoint_op(self, kspace):
        # A^H(y) = S^H * F^-1(M * y)
        # M * y
        masked_kspace = kspace * self.mask
        # F^-1(...)
        image_coils = torch.fft.ifftn(masked_kspace, dim=(-3, -2, -1), norm="ortho")
        # S^H * ...
        image = torch.sum(image_coils * self.maps.conj(), dim=1, keepdim=True)
        return image

    def normal(self, x):
        # A^H A x + lambda x
        out = self.adjoint_op(self.forward_op(x))
        if self.l2lam > 0:
            out = out + self.l2lam * x
        return out

def dot_batch(x1, x2):
    # inner product for batched complex tensors
    return torch.sum(x1.conj() * x2, dim=list(range(1, len(x1.shape)))).real.view(-1, *(1,)*(len(x1.shape)-1))

def ip_batch(x):
    return dot_batch(x, x)

def conjgrad(x, b, Aop_fun, max_iter=10, eps=1e-4):
    """
    Differentiable Conjugate Gradient solver for complex tensors.
    """
    r = b - Aop_fun(x)
    p = r.clone()

    rsnot = ip_batch(r)
    rsold = rsnot.clone()
    rsnew = rsnot.clone()

    for i in range(max_iter):
        if rsnew.max() < eps:
            break

        Ap = Aop_fun(p)
        pAp = dot_batch(p, Ap)

        alpha = rsold / pAp
        x = x + alpha * p
        r = r - alpha * Ap

        rsnew = ip_batch(r)
        beta = rsnew / rsold
        rsold = rsnew.clone()

        p = beta * p + r

    return x

class UnrolledModel3D(nn.Module):
    """
    3D Unrolled Compressed Sensing for Precision Brain Morphometry.
    """
    def __init__(self, num_grad_steps=5, num_cg_steps=10, modl_lamda=0.05):
        super().__init__()
        
        self.num_grad_steps = num_grad_steps
        self.num_cg_steps = num_cg_steps
        self.modl_lamda = modl_lamda

        self.resnets = nn.ModuleList([
            ResNet(num_resblocks=5, in_chans=2, chans=32, kernel_size=3, drop_prob=0.0, circular_pad=False)
            for _ in range(self.num_grad_steps)
        ])

    def forward(self, kspace, maps, mask):
        A = SenseModel_3D_Harvard(maps, mask, l2lam=self.modl_lamda)
        image = A.adjoint_op(kspace)
        
        for resnet in self.resnets:
            image_real = image.real
            image_imag = image.imag
            image_2ch = torch.cat([image_real, image_imag], dim=1)
            
            image_2ch = resnet(image_2ch)
            image_reg = torch.complex(image_2ch[:, 0:1, ...], image_2ch[:, 1:2, ...])
            
            rhs = A.adjoint_op(kspace) + self.modl_lamda * image_reg
            image = conjgrad(x=image, b=rhs, Aop_fun=A.normal, max_iter=self.num_cg_steps)
            
        return image
