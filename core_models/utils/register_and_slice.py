import SimpleITK as sitk
import numpy as np
import os

def _load_image(input_data):
    """Helper to flexibly load a string path (.nii.gz, .npy) or an existing object into a sitk.Image"""
    if isinstance(input_data, sitk.Image):
        return input_data
    elif isinstance(input_data, np.ndarray):
        img = sitk.GetImageFromArray(input_data.astype(np.float32))
        img.SetSpacing([1.0, 1.0, 1.0])
        return img
    elif isinstance(input_data, str):
        if input_data.endswith('.npy'):
            arr = np.load(input_data)
            img = sitk.GetImageFromArray(arr.astype(np.float32))
            img.SetSpacing([1.0, 1.0, 1.0])
            return img
        else:
            return sitk.ReadImage(input_data, sitk.sitkFloat32)
    else:
        raise ValueError(f"Unsupported input type: {type(input_data)}")

def register_and_slice_block(adni_input, target_cs_input, block_cs_inputs, block_name, output_dir):
    """Registers ADNI to a target CS scan and extracts slices for a block of scans."""
    print(f"\n--- Processing {block_name} ---")
    
    # 1. Load the images flexibly
    adni_volume = _load_image(adni_input)
    fixed_image = _load_image(target_cs_input)
    
    # 2. Set up the Rigid Registration
    # Switch to GEOMETRY instead of MOMENTS because background noise throws off the center-of-mass calculation
    initial_transform = sitk.CenteredTransformInitializer(
        fixed_image, adni_volume, 
        sitk.Euler3DTransform(), 
        sitk.CenteredTransformInitializerFilter.GEOMETRY
    )

    registration_method = sitk.ImageRegistrationMethod()
    # Use MeanSquares since the images are identical except for noise/rotation
    registration_method.SetMetricAsMeanSquares()
    # Use 100% of the pixels to completely average out the severe random noise
    registration_method.SetMetricSamplingStrategy(registration_method.NONE)
    registration_method.SetInterpolator(sitk.sitkLinear)
    
    # Multi-resolution framework for robustness
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    registration_method.SetOptimizerAsRegularStepGradientDescent(
        learningRate=2.0,
        minStep=0.001,
        numberOfIterations=200,
        relaxationFactor=0.5
    )
    registration_method.SetOptimizerScalesFromPhysicalShift()
    registration_method.SetInitialTransform(initial_transform, inPlace=False)

    # 3. Execute Registration
    target_name = os.path.basename(target_cs_input) if isinstance(target_cs_input, str) else block_name
    print(f"Aligning ADNI to target: {target_name}...")
    final_transform = registration_method.Execute(fixed_image, adni_volume)

    # 4. Resample the ADNI image into the target's grid
    resampled_adni = sitk.Resample(
        adni_volume, fixed_image, 
        final_transform, sitk.sitkLinear, 0.0, adni_volume.GetPixelID()
    )
    
    # Optional: Save the 3D resampled ADNI for visual verification later
    np.save(os.path.join(output_dir, f"{block_name}_resampled_ADNI.npy"), sitk.GetArrayFromImage(resampled_adni))

    # 5. Extract Slices for all CS scans in this block
    adni_array = sitk.GetArrayFromImage(resampled_adni)
    
    for i, cs_input in enumerate(block_cs_inputs):
        cs_image = _load_image(cs_input)
        cs_array = sitk.GetArrayFromImage(cs_image)
        
        if isinstance(cs_input, str):
            scan_name = os.path.basename(cs_input).split('.')[0]
        else:
            scan_name = f"{block_name}_scan_{i}"
        
        num_slices = cs_array.shape[0]
        for i in range(num_slices):
            # Extract 2D coronal slices
            cs_slice = cs_array[i, :, :]
            adni_slice = adni_array[i, :, :]
            
            # Save the pairs
            np.save(os.path.join(output_dir, f"{scan_name}_slice_{i:03d}_input.npy"), cs_slice)
            np.save(os.path.join(output_dir, f"{scan_name}_slice_{i:03d}_target.npy"), adni_slice)
            
        print(f"Extracted {num_slices} slices for {scan_name}")

def process_session(adni_path, cs_scan_paths, output_dir):
    """Manages the Block A and Block B logic for a single session."""
    os.makedirs(output_dir, exist_ok=True)
    adni_volume = sitk.ReadImage(adni_path, sitk.sitkFloat32)
    
    # Block A: Scans 1-4. Target is Scan 1.
    block_a_paths = cs_scan_paths[0:4]
    register_and_slice_block(adni_volume, block_a_paths[0], block_a_paths, "Block_A", output_dir)
    
    # Block B: Scans 5-8. Target is Scan 5.
    block_b_paths = cs_scan_paths[4:8]
    register_and_slice_block(adni_volume, block_b_paths[0], block_b_paths, "Block_B", output_dir)

# --- Example Usage ---
# cs_files = [
#     "scan1.nii.gz", "scan2.nii.gz", "scan3.nii.gz", "scan4.nii.gz", 
#     "scan5.nii.gz", "scan6.nii.gz", "scan7.nii.gz", "scan8.nii.gz"
# ]
# process_session("adni_baseline.nii.gz", cs_files, "./training_data/")