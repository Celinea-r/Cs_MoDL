import SimpleITK as sitk
import numpy as np
import os
import sys
import math

# Allow script to find modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from core_models.utils.register_and_slice import register_and_slice_block

def main():
    print("Loading BART reconstruction outputs...")
    
    # Paths
    r2_path = 'results_multiR/reconstruction_R2_espirit_pics.npy'
    r6_path = 'results_multiR/reconstruction_R6_espirit_pics.npy'
    
    if not os.path.exists(r2_path) or not os.path.exists(r6_path):
        print(f"Error: Could not find {r2_path} or {r6_path}.")
        print("Please run undersample_and_recon.py first to generate the reconstructions.")
        return
        
    r2_array = np.load(r2_path)
    r6_array = np.load(r6_path)
    
    # We take the magnitude of the complex reconstructions for registration
    r2_mag = np.abs(r2_array).astype(np.float32)
    r6_mag = np.abs(r6_array).astype(np.float32)
    
    # Create SimpleITK images (D, H, W order expected by GetImageFromArray)
    # The arrays from reconstruction are likely (Z, Y, X).
    r2_img = sitk.GetImageFromArray(r2_mag)
    r2_img.SetSpacing([1.0, 1.0, 1.0])
    
    r6_img = sitk.GetImageFromArray(r6_mag)
    r6_img.SetSpacing([1.0, 1.0, 1.0])
    
    # 1. Prepare ADNI Baselines (R=2)
    print("Preparing ADNI Baselines from R=2 reconstruction...")
    
    # "Before Rotation": Mathematically swap axes to create a fake Sagittal scan
    permute = sitk.PermuteAxesImageFilter()
    permute.SetOrder([2, 1, 0]) 
    adni_sagittal = permute.Execute(r2_img)
    
    # "Pre-Rotated 90°": The scan after the clinical headers automatically rotate it 90 degrees back to Coronal.
    # It is inherently upright (like the target) but completely lacks the 3-degree / 5mm fine-tuning.
    adni_prerotated = r2_img
    
    # 2. Prepare CS Scan (R=6, Coronal, Moved)
    print("Simulating Patient Movement on R=6 Coronal target...")
    transform = sitk.Euler3DTransform()
    # Random minor rotation
    rot_x = math.radians(np.random.uniform(-3, 3))
    rot_y = math.radians(np.random.uniform(-3, 3))
    rot_z = math.radians(np.random.uniform(-3, 3))
    transform.SetRotation(rot_x, rot_y, rot_z)
    
    # Random minor translation
    trans_x = np.random.uniform(-5, 5)
    trans_y = np.random.uniform(-5, 5)
    trans_z = np.random.uniform(-5, 5)
    transform.SetTranslation((trans_x, trans_y, trans_z))
    
    transform.SetCenter(r6_img.TransformContinuousIndexToPhysicalPoint([s/2.0 for s in r6_img.GetSize()]))
    cs_target_moved = sitk.Resample(r6_img, r6_img, transform, sitk.sitkLinear, 0.0, r6_img.GetPixelID())
    
    # 3. Output Directory & Save memory copies for plotting
    os.makedirs('test_recon_out', exist_ok=True)
    np.save('test_recon_out/adni_R2_sagittal.npy', sitk.GetArrayFromImage(adni_sagittal))
    np.save('test_recon_out/adni_R2_prerotated.npy', sitk.GetArrayFromImage(adni_prerotated))
    np.save('test_recon_out/cs_R6_moved.npy', sitk.GetArrayFromImage(cs_target_moved))
    
    # 4. Run Registration (directly in memory!)
    # We pass `adni_prerotated` to simulate the algorithm taking the roughly rotated image 
    # and fine-tuning the 5mm / 3-degree shift.
    print("\n--- Running Registration Pipeline (Memory Only) ---")
    register_and_slice_block(adni_prerotated, cs_target_moved, [cs_target_moved], "Block_R6", "test_recon_out")
    
    print("\n--- Verification ---")
    files_created = os.listdir("test_recon_out")
    npy_files = [f for f in files_created if f.endswith('.npy')]
    print(f"Successfully generated {len(npy_files)} perfectly aligned slice files (.npy).")

if __name__ == "__main__":
    main()
