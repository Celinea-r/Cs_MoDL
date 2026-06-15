import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
import os

def get_slice(image, z_slice):
    arr = sitk.GetArrayFromImage(image)
    return arr[z_slice, :, :]

# Load the volumes
cs_1 = sitk.ReadImage('test_out/cs_scan_1_target.nii.gz')
cs_2 = sitk.ReadImage('test_out/cs_scan_2_target.nii.gz')
adni_baseline = sitk.ReadImage('test_out/adni_baseline.nii.gz')
block_a_resampled = sitk.ReadImage('test_out/Block_A_resampled_ADNI.nii.gz')
block_b_resampled = sitk.ReadImage('test_out/Block_B_resampled_ADNI.nii.gz')

# Pick 3 slices perfectly spaced apart across the entire brain (25%, 50%, 75% depth)
max_z = cs_1.GetSize()[2]
slices_to_plot = [max_z // 4, max_z // 2, (max_z * 3) // 4]

os.makedirs('outputs', exist_ok=True)

# ---------------------------------------------------------
# Plot 1: Scan 1 (Pre-Break)
# ---------------------------------------------------------
fig1, axes1 = plt.subplots(3, 3, figsize=(12, 12))
fig1.suptitle("Registration Results: CS Scan 1 (Pre-Break)", fontsize=16)

for i, z_slice in enumerate(slices_to_plot):
    axes1[i, 0].imshow(get_slice(adni_baseline, z_slice), cmap='gray')
    if i == 0: axes1[i, 0].set_title('ADNI Baseline')
    axes1[i, 0].set_ylabel(f'Slice {z_slice}')
    
    axes1[i, 1].imshow(get_slice(cs_1, z_slice), cmap='gray')
    if i == 0: axes1[i, 1].set_title('CS Scan 1 Target (Noisy)')
    
    axes1[i, 2].imshow(get_slice(block_a_resampled, z_slice), cmap='gray')
    if i == 0: axes1[i, 2].set_title('Registered ADNI Result')

for ax in axes1.flatten():
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
out1 = 'outputs/registration_scan1.png'
fig1.savefig(out1, dpi=150)
print(f"Saved Pre-Break plot to {out1}")
plt.close(fig1)

# ---------------------------------------------------------
# Plot 2: Scan 2 (Post-Break)
# ---------------------------------------------------------
fig2, axes2 = plt.subplots(3, 3, figsize=(12, 12))
fig2.suptitle("Registration Results: CS Scan 2 (Post-Break, Moved)", fontsize=16)

for i, z_slice in enumerate(slices_to_plot):
    axes2[i, 0].imshow(get_slice(adni_baseline, z_slice), cmap='gray')
    if i == 0: axes2[i, 0].set_title('ADNI Baseline')
    axes2[i, 0].set_ylabel(f'Slice {z_slice}')
    
    axes2[i, 1].imshow(get_slice(cs_2, z_slice), cmap='gray')
    if i == 0: axes2[i, 1].set_title('CS Scan 2 Target (Moved & Noisy)')
    
    axes2[i, 2].imshow(get_slice(block_b_resampled, z_slice), cmap='gray')
    if i == 0: axes2[i, 2].set_title('Registered ADNI Result')

for ax in axes2.flatten():
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
out2 = 'outputs/registration_scan2.png'
fig2.savefig(out2, dpi=150)
print(f"Saved Post-Break plot to {out2}")
plt.close(fig2)
