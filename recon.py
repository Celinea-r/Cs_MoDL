import h5py
import numpy as np
import matplotlib.pyplot as plt


filename = "raw_data/train_R_1/e14191s3_P58368.7.h5"


def fftc(x, axes):
    """Centered orthonormal FFT."""
    return np.fft.fftshift(
        np.fft.fftn(
            np.fft.ifftshift(x, axes=axes),
            axes=axes,
            norm="ortho",
        ),
        axes=axes,
    )


def ifftc(x, axes):
    """Centered orthonormal inverse FFT."""
    return np.fft.fftshift(
        np.fft.ifftn(
            np.fft.ifftshift(x, axes=axes),
            axes=axes,
            norm="ortho",
        ),
        axes=axes,
    )


# --------------------------------------------------
# 1. Load Calgary hybrid k-space
#    raw shape: (256, 218, 170, 24)
# --------------------------------------------------
with h5py.File(filename, "r") as f:
    raw = f["kspace"][()]

print("Raw shape:", raw.shape)


# --------------------------------------------------
# 2. Convert interleaved real/imaginary channels
#    Result: (256, 218, 170, 12)
# --------------------------------------------------
hybrid = (
    raw[..., 0::2].astype(np.float32)
    + 1j * raw[..., 1::2].astype(np.float32)
)

print("Complex hybrid shape:", hybrid.shape)


# --------------------------------------------------
# 3. Rearrange into:
#    (coil, ky, kz, x)
#
#    Original:
#    (x, ky, kz, coil)
# --------------------------------------------------
hybrid = np.transpose(hybrid, (3, 1, 2, 0))

print("Rearranged shape:", hybrid.shape)
# Expected: (12, 218, 170, 256)


# --------------------------------------------------
# 4. Optional but recommended:
#    explicitly zero the unacquired partial-Fourier tail
#
#    For Nz=170, the dataset documentation says that
#    approximately the first 145 locations were acquired.
# --------------------------------------------------
if hybrid.shape[2] == 170:
    hybrid[:, :, 145:, :] = 0

elif hybrid.shape[2] == 174:
    hybrid[:, :, 148:, :] = 0

elif hybrid.shape[2] == 180:
    hybrid[:, :, 153:, :] = 0


# --------------------------------------------------
# 5. The last axis is already in image space.
#    Transform it back to Fourier space first.
# --------------------------------------------------
kspace_3d = fftc(hybrid, axes=(3,))


# --------------------------------------------------
# 6. Perform centered 3D inverse FFT
#
#    Spatial dimensions are now:
#    axis 1 = 218
#    axis 2 = 170
#    axis 3 = 256
# --------------------------------------------------
coil_images = ifftc(kspace_3d, axes=(1, 2, 3))


# --------------------------------------------------
# 7. Calgary-specific rearrangement needed to obtain
#    a coherent spatial volume
# --------------------------------------------------
coil_images = np.fft.ifftshift(
    coil_images,
    axes=(1, 2),
)


# --------------------------------------------------
# 8. Root-sum-of-squares across coils
#    Result: (218, 170, 256)
# --------------------------------------------------
rss = np.sqrt(np.sum(np.abs(coil_images) ** 2, axis=0))

print("RSS shape:", rss.shape)


# --------------------------------------------------
# 9. Display three orthogonal central slices
# --------------------------------------------------

# Axial-type view
plt.figure(figsize=(6, 6))
plt.imshow(
    rss[:, :, rss.shape[2] // 2].T,
    cmap="gray",
    origin="lower",
)
plt.title("Central slice along axis 2")
plt.axis("off")
plt.tight_layout()
plt.show()


# Sagittal-type view
plt.figure(figsize=(6, 6))
plt.imshow(
    rss[:, rss.shape[1] // 2, :].T,
    cmap="gray",
    origin="lower",
)
plt.title("Central slice along axis 1")
plt.axis("off")
plt.tight_layout()
plt.show()


# Coronal-type view
plt.figure(figsize=(6, 6))
plt.imshow(
    rss[rss.shape[0] // 2, :, :].T,
    cmap="gray",
    origin="lower",
)
plt.title("Central slice along axis 0")
plt.axis("off")
plt.tight_layout()
plt.show()


'''
with h5py.File(filename, "r") as f:
    raw = f["kspace"][()]

print("Raw shape:", raw.shape)

# Correct Calgary representation:
# channel 0 = real coil 1
# channel 1 = imag coil 1
# channel 2 = real coil 2
# channel 3 = imag coil 2, etc.
kspace = raw[..., 0::2] + 1j * raw[..., 1::2]

print("Complex k-space shape:", kspace.shape)
# Expected: (256, 218, 170, 12)
# kspace shape: (256, 218, 170, 12)

# A ky-kz point is considered sampled if any readout position
# or coil contains a nonzero value.
mask = np.any(np.abs(kspace) > 0, axis=(0, 3))

print("Mask shape:", mask.shape)
print("Sampled fraction:", mask.mean())
print("Estimated acceleration:", 1.0 / mask.mean())

# Centered inverse FFT over the two encoded dimensions
coil_images = np.fft.fftshift(
    np.fft.ifft2(
        np.fft.ifftshift(kspace, axes=(1, 2)),
        axes=(1, 2),
        norm="ortho",
    ),
    axes=(1, 2),
)

# Root-sum-of-squares over the 12 coils
rss = np.sqrt(np.sum(np.abs(coil_images) ** 2, axis=3))

print("RSS shape:", rss.shape)
# Expected: (256, 218, 170)

slice_idx = rss.shape[0] // 2

plt.figure(figsize=(6, 6))
plt.imshow(
    rss[slice_idx, :, :].T,
    cmap="gray",
    origin="lower",
)
plt.title(f"Plane {slice_idx}")
plt.axis("off")
plt.tight_layout()
plt.show()

'''
'''
import h5py
import numpy as np
import matplotlib.pyplot as plt

filename = "raw_data/test_R_5/e13991s3_P01536.7.h5"

with h5py.File(filename, "r") as f:
    print("Top-level keys:", list(f.keys()))

    def show_dataset(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(name, obj.shape, obj.dtype)

    f.visititems(show_dataset)

with h5py.File(filename, "r") as f:
    raw = f["kspace"][()]

print(raw.shape, raw.dtype)

# Possibility 1: first 12 real, last 12 imaginary
kspace_split = raw[..., :12] + 1j * raw[..., 12:]

# Possibility 2: interleaved real/imag
kspace_interleaved = raw[..., 0::2] + 1j * raw[..., 1::2]

print("Split shape:", kspace_split.shape)
print("Interleaved shape:", kspace_interleaved.shape)

def reconstruct_rss(kspace):
    # Stored data are x-ky-kz, so only inverse FFT ky and kz
    coil_images = np.fft.ifftshift(kspace, axes=(1, 2))
    coil_images = np.fft.ifftn(
        coil_images,
        axes=(1, 2),
        norm="ortho",
    )
    coil_images = np.fft.fftshift(coil_images, axes=(1, 2))

    rss = np.sqrt(np.sum(np.abs(coil_images) ** 2, axis=3))
    return rss


rss_split = reconstruct_rss(kspace_split)
rss_interleaved = reconstruct_rss(kspace_interleaved)

slice_idx = rss_split.shape[2] // 2

plt.figure()
plt.imshow(rss_split[:, :, slice_idx].T, cmap="gray", origin="lower")
plt.title("First 12 real, last 12 imaginary")
plt.axis("off")
plt.show()

plt.figure()
plt.imshow(rss_interleaved[:, :, slice_idx].T, cmap="gray", origin="lower")
plt.title("Interleaved real/imag")
plt.axis("off")
plt.show()

'''