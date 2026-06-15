import matplotlib.pyplot as plt
import numpy as np
import os

def get_slice(arr, z_slice):
    return arr[z_slice, :, :]

def normalize(img):
    img = img - np.min(img)
    if np.max(img) > 0:
        img = img / np.max(img)
    return img

adni_sagittal = np.load('test_recon_out/adni_R2_sagittal.npy')
adni_prerotated = np.load('test_recon_out/adni_R2_prerotated.npy')
cs_target = np.load('test_recon_out/cs_R6_moved.npy')
block_resampled = np.load('test_recon_out/Block_R6_resampled_ADNI.npy')

max_z = cs_target.shape[0]
slices_to_plot = [
    int(max_z * 0.20),
    int(max_z * 0.35),
    int(max_z * 0.50),
    int(max_z * 0.65),
    int(max_z * 0.80)
]

os.makedirs('outputs', exist_ok=True)

fig, axes = plt.subplots(5, 4, figsize=(20, 25))
fig.suptitle("Registration: Sagittal R=2 ADNI to Moved Coronal R=6 CS Target", fontsize=22)

for i, z_slice in enumerate(slices_to_plot):
    sag_slice = get_slice(adni_sagittal, z_slice)
    pre_slice = get_slice(adni_prerotated, z_slice)
    cs_slice = get_slice(cs_target, z_slice)
    res_slice = get_slice(block_resampled, z_slice)
    
    # Apply flipud to display upright
    axes[i, 0].imshow(np.flipud(normalize(sag_slice)), cmap='gray')
    if i == 0: axes[i, 0].set_title('ADNI Baseline\n(R=2, Sagittal, Before Rotation)', fontsize=14)
    axes[i, 0].set_ylabel(f'Slice {z_slice}', fontsize=12)
    
    axes[i, 1].imshow(np.flipud(normalize(pre_slice)), cmap='gray')
    if i == 0: axes[i, 1].set_title('ADNI Rotated, Pre-Registered\n(R=2, Coronal, Misaligned)', fontsize=14)

    axes[i, 2].imshow(np.flipud(normalize(res_slice)), cmap='gray')
    if i == 0: axes[i, 2].set_title('Registered ADNI\n(Perfectly Aligned Coronal)', fontsize=14)
    
    axes[i, 3].imshow(np.flipud(normalize(cs_slice)), cmap='gray')
    if i == 0: axes[i, 3].set_title('CS Target\n(R=6, Coronal, Moved)', fontsize=14)

for ax in axes.flatten():
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
out_path = 'outputs/registration_recon_plot.png'
fig.savefig(out_path, dpi=150)
print(f"Saved plot to {out_path}")
plt.close(fig)
