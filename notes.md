# Registration Pipeline Concepts

## 1. The Conceptual Framework

*   **Target (Fixed):** Conceptually, this is your **CS-reconstructed scan** (and eventually, your MoDL scan). Think of this as your "anchor." It stays completely still. Whatever orientation the patient's head was in during the CS scan, this defines the grid we want to use.
*   **Rotated ADNI (Moving):** This represents the high-quality baseline ADNI scan you downloaded for the patient. Because the ADNI scan was taken on a different day or scanner, the patient's head was likely tilted, shifted, or rotated differently than it was during the CS scan. 
*   **Registered ADNI (Result):** This is the ADNI scan *after* the algorithm processes it. The goal is that this result perfectly mimics the pose, grid, and geometry of the Target (CS scan). We need them perfectly aligned so that when you feed slice #42 of the CS scan into your network, slice #42 of the ADNI scan matches the exact same physical piece of brain tissue for calculating the Loss.

## 2. How Exactly Are We Rotating It?

We aren't manually telling the computer "rotate it X degrees." The script uses a mathematical process called **Rigid Image Registration**. Here is how the SimpleITK code actually works:

1.  **The Transform (`Euler3DTransform`)**: We tell the computer, "You are allowed to physically shift the brain along X, Y, and Z axes, and you are allowed to rotate it (pitch, roll, yaw), but you *cannot* stretch or deform the brain."
2.  **The Metric (`Correlation`)**: The computer overlays the ADNI image on top of the Target image and multiplies their pixel intensities together to calculate a "Correlation Score." If the brains don't line up, the score is very low.
3.  **The Optimizer (`RegularStepGradientDescent`)**: This is the engine. It tests a tiny rotation, checks the Correlation score, and sees if the score went up or down. If the score went up, it rotates a little more in that direction. It calculates the mathematical gradient (the slope) and "walks down the hill" step-by-step.
4.  **Multi-Resolution Pyramid**: Because a full-resolution 3D brain is incredibly complex (millions of pixels), the optimizer can get confused. So, we told it to shrink both brains down by 4x, find the rough rotation, then double the resolution and fine-tune it, and finally use the full-resolution images for the last micro-adjustments.
5.  **Resampling**: Once the optimizer finds the mathematical "winning angle" where the Correlation score is highest, it uses the `sitk.Resample()` function. This permanently applies the winning 3D mathematical matrix to the ADNI volume, rotating every single pixel to match the Target voxel-for-voxel.
