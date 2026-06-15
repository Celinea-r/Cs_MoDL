# The CS-MoDL System Pipeline

This diagram illustrates the complete, end-to-end data flow for the Harvard dataset, specifically accounting for patient movement (the Pre/Post Break paradigm) and the 2D Multi-Coil MoDL reconstruction.

```mermaid
graph TD
    %% Phase 1: Offline Registration
    subgraph phase1 ["Phase 1: Offline Preprocessing (register_and_slice.py)"]
        direction TB
        A["Sagittal ADNI Baseline"]
        
        subgraph cs_scans ["Coronal CS Scans"]
            direction TB
            B1["CS Scans 1-4 (Pre-Break)"]
            B2["CS Scans 5-8 (Post-Break / Moved)"]
        end

        A --> |"GEOMETRY Initializer"| Reg1{"Rigid Registration"}
        B1 --> |"Target"| Reg1
        Reg1 --> |"MeanSquares Optimizer"| Slice1["Extract 2D Slices"]
        
        A --> |"GEOMETRY Initializer"| Reg2{"Rigid Registration"}
        B2 --> |"Target (Rotated/Shifted)"| Reg2
        Reg2 --> |"MeanSquares Optimizer"| Slice2["Extract 2D Slices"]
        
        Slice1 --> C[/"Perfectly Aligned 2D .npy Ground Truths"/]
        Slice2 --> C
    end

    %% Phase 2: Dataloader
    subgraph phase2 ["Phase 2: Online Training Data (core_models/utils/dataset.py)"]
        direction TB
        C -->|"Load Target"| D["PyTorch HarvardKSpaceDataset"]
        RawK["Harvard Raw K-Space (.npy)"] -->|"Load Inputs"| D
        RawM["ESPIRiT Sensitivity Maps"] --> D
        RawMask["Sampling Masks"] --> D
        
        D --> E[/"Batched Tensors: [B, C, H, W]"/]
    end

    %% Phase 3: MoDL Neural Network
    subgraph phase3 ["Phase 3: 2D Unrolled MoDL Network"]
        direction TB
        E --> F{"Unrolled Iterations (e.g. 5 steps)"}
        F -->|"1. UNet Regularizer"| G["Denoised Image Tensor"]
        G -->|"2. Differentiable CG Solver"| H["Physics-Enforced Image"]
        H --> F
        H -->|"Final Iteration"| I["Final Reconstructed 2D Image"]
    end

    %% Phase 4: Loss
    subgraph phase4 ["Phase 4: Optimization"]
        direction TB
        I --> J("Hybrid Loss Function")
        C -.->|"Target"| J
        J -->|"L1 Loss Component"| K("Pixel-Level Accuracy")
        J -->|"2D SSIM Component"| L("Structural Similarity")
        K --> M["Adam Optimizer: Update UNet Weights"]
        L --> M
    end
```

### Detailed Breakdown

#### Phase 1: Offline Preprocessing (Handling the Paradigm)
The `register_and_slice.py` script runs **once** offline. Because the patient leaves the scanner during their break, their head orientation completely changes between `CS Scan 4` and `CS Scan 5`. 
To solve this, the script runs the registration algorithm independently for each Block. It uses the `GEOMETRY` initializer to mathematically map the Sagittal ADNI scan to the Coronal CS space. The robust `MeanSquares` optimizer pierces through the noise and locks onto the exact rotation/shift of the patient's head for that specific scan. Finally, it slices the 3D volumes into 2D pieces, saving thousands of `.npy` arrays to the hard drive.

#### Phase 2: The PyTorch DataLoader
The neural network does not waste time doing registration! During training, `core_models/utils/dataset.py` acts as a high-speed engine. It grabs the complex raw `K-Space`, the `Sensitivity Maps`, the `Sampling Mask`, and the pre-aligned `Target` from the hard drive, bundles them into PyTorch Tensors, and fires them into the GPU.

#### Phase 3: The MoDL Architecture
The `UnrolledModel2D` mimics the physics of an MRI machine. 
1. The **UNet Regularizer** looks at the noisy image and removes artifacts.
2. The **Differentiable CG Solver** takes that denoised image, checks it against the raw `K-Space` and `Sensitivity Maps`, and mathematically forces the image to obey the physics of the MRI coils. 
It loops through this process several times (unrolling), refining the image at each step.

#### Phase 4: Hybrid Loss
The final image is compared to the perfectly aligned Target. Instead of just checking if the pixels are the right color (L1 Loss), it also uses **2D SSIM** to evaluate the structural integrity of the brain folds and tissues, ensuring a hyper-realistic reconstruction!
