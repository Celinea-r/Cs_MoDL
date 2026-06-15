import numpy as np
import os

def create_mock_data(output_dir='harvard_mock_data', num_samples=10):
    os.makedirs(output_dir, exist_ok=True)
    
    # B, C, H, W -> Batch is handled by Dataloader. So arrays are [C, H, W]
    C, H, W = 4, 128, 128
    
    for i in range(num_samples):
        # Fake K-space (Complex)
        kspace = np.random.randn(C, H, W) + 1j * np.random.randn(C, H, W)
        kspace = kspace.astype(np.complex64)
        
        # Fake ESPIRiT maps (Complex)
        maps = np.random.randn(C, H, W) + 1j * np.random.randn(C, H, W)
        maps = maps.astype(np.complex64)
        
        # Fake Mask (Float)
        mask = (np.random.rand(1, H, W) > 0.5).astype(np.float32)
        
        # Fake Target (Complex) - Target is 1 channel (the combined image)
        target = np.random.randn(1, H, W) + 1j * np.random.randn(1, H, W)
        target = target.astype(np.complex64)
        
        np.save(os.path.join(output_dir, f"sample_{i:04d}_kspace.npy"), kspace)
        np.save(os.path.join(output_dir, f"sample_{i:04d}_maps.npy"), maps)
        np.save(os.path.join(output_dir, f"sample_{i:04d}_mask.npy"), mask)
        np.save(os.path.join(output_dir, f"sample_{i:04d}_target.npy"), target)
        
    print(f"Created {num_samples} mock Harvard data samples in {output_dir}")

if __name__ == "__main__":
    create_mock_data()
