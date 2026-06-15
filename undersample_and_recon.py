#!/usr/bin/env python3

"""
Retrospective R=2 undersampling and reconstruction of one
Calgary-Campinas 12-channel fully sampled HDF5 scan.

Outputs:
    - Fully sampled RSS reconstruction
    - Zero-filled R=2 RSS reconstruction
    - ESPIRiT/PICS R=2 reconstruction using BART
    - Sampling mask
    - Three-way comparison figure
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# BART import
# ============================================================

def import_bart():
    """
    Import the official BART Python wrapper.

    BART should be installed from:
        ~/bart/pkg/python

    using:
        cd ~/bart/pkg/python
        pip install .
    """

    bart_path = Path.home() / "bart"

    os.environ.setdefault(
        "BART_TOOLBOX_PATH",
        str(bart_path),
    )

    try:
        from bart import bart
    except ImportError as exc:
        raise ImportError(
            "\nCould not import the BART Python wrapper.\n\n"
            "Install it using:\n"
            "    cd ~/bart/pkg/python\n"
            "    python -m pip install .\n\n"
            "Then verify:\n"
            "    python -c \"from bart import bart; "
            "bart(0, 'version')\"\n"
        ) from exc

    return bart


# ============================================================
# Fourier-transform helpers
# ============================================================

def fftc(
    data: np.ndarray,
    axes: tuple[int, ...],
) -> np.ndarray:
    """Centered orthonormal multidimensional FFT."""

    return np.fft.fftshift(
        np.fft.fftn(
            np.fft.ifftshift(data, axes=axes),
            axes=axes,
            norm="ortho",
        ),
        axes=axes,
    )


def ifftc(
    data: np.ndarray,
    axes: tuple[int, ...],
) -> np.ndarray:
    """Centered orthonormal multidimensional inverse FFT."""

    return np.fft.fftshift(
        np.fft.ifftn(
            np.fft.ifftshift(data, axes=axes),
            axes=axes,
            norm="ortho",
        ),
        axes=axes,
    )


# ============================================================
# Calgary data loading
# ============================================================

def load_calgary_hybrid(
    filename: Path,
) -> np.ndarray:
    """
    Load Calgary-Campinas hybrid-space data.

    Stored HDF5 shape:
        (x, ky, kz, 2 * coils)

    The last dimension contains interleaved components:
        real coil 1
        imaginary coil 1
        real coil 2
        imaginary coil 2
        ...

    Returned shape:
        (coil, ky, kz, x)
    """

    if not filename.exists():
        raise FileNotFoundError(
            f"Input file does not exist: {filename}"
        )

    print(f"Loading: {filename}")

    with h5py.File(filename, "r") as file:
        if "kspace" not in file:
            raise KeyError(
                f"'kspace' was not found. Keys: {list(file.keys())}"
            )

        raw = file["kspace"][()]

    print("Raw shape:", raw.shape)
    print("Raw dtype:", raw.dtype)

    if raw.ndim != 4:
        raise ValueError(
            f"Expected a 4D array, received shape {raw.shape}"
        )

    if raw.shape[-1] % 2 != 0:
        raise ValueError(
            "The last dimension must contain real/imaginary pairs."
        )

    number_of_coils = raw.shape[-1] // 2

    # Interleaved real and imaginary components.
    hybrid = (
        raw[..., 0::2].astype(np.float32)
        + 1j * raw[..., 1::2].astype(np.float32)
    )

    # Original:
    #     (x, ky, kz, coil)
    #
    # Reordered:
    #     (coil, ky, kz, x)
    hybrid = np.transpose(
        hybrid,
        (3, 1, 2, 0),
    )

    hybrid = hybrid.astype(np.complex64)

    print("Number of coils:", number_of_coils)
    print("Complex hybrid shape:", hybrid.shape)

    return hybrid


def zero_partial_fourier_tail(
    hybrid: np.ndarray,
) -> np.ndarray:
    """
    Explicitly zero the nominally unacquired Calgary kz tail.

    Input shape:
        (coil, ky, kz, x)
    """

    result = hybrid.copy()

    kz_size = result.shape[2]

    known_cutoffs = {
        170: 145,
        174: 148,
        180: 153,
    }

    cutoff = known_cutoffs.get(kz_size)

    if cutoff is None:
        print(
            f"No predefined partial-Fourier cutoff for kz={kz_size}. "
            "No explicit tail removal was applied."
        )
        return result

    result[:, :, cutoff:, :] = 0

    print(
        "Zeroed nominal partial-Fourier tail:",
        f"kz indices {cutoff}:{kz_size}",
    )

    return result


def hybrid_to_full_kspace(
    hybrid: np.ndarray,
) -> np.ndarray:
    """
    Convert Calgary hybrid-space data into 3D k-space.

    Calgary shape:
        (coil, ky, kz, x)

    The final x dimension is stored in image space, so a centered
    FFT is applied along that dimension.

    Returned shape:
        (coil, ky, kz, kx)
    """

    full_kspace = fftc(
        hybrid,
        axes=(3,),
    )

    return full_kspace.astype(np.complex64)


# ============================================================
# Sampling mask
# ============================================================

def make_regular_mask(
    ky_size: int,
    kz_size: int,
    kx_size: int,
    acs_lines: int,
    offset: int,
    acceleration: int,
) -> np.ndarray:
    """
    Generate a regular Cartesian mask along ky.

    Every second ky line is retained. A central fully sampled
    ACS region is added for ESPIRiT calibration.

    Returned shape:
        (1, ky, kz, kx)
    """

    if offset not in (0, 1):
        raise ValueError("offset must be 0 or 1.")

    if acs_lines < 1:
        raise ValueError(
            "acs_lines must be positive for ESPIRiT calibration."
        )

    if acs_lines > ky_size:
        raise ValueError(
            f"acs_lines={acs_lines} exceeds ky size {ky_size}."
        )

    ky_mask = np.zeros(
        ky_size,
        dtype=np.float32,
    )

    # Regular sampling.
    ky_mask[offset::acceleration] = 1.0

    # Fully sampled central ACS lines.
    center = ky_size // 2
    start = center - acs_lines // 2
    end = start + acs_lines

    ky_mask[start:end] = 1.0

    mask = ky_mask[
        np.newaxis,
        :,
        np.newaxis,
        np.newaxis,
    ]

    mask = np.broadcast_to(
        mask,
        (1, ky_size, kz_size, kx_size),
    ).copy()

    return mask


def calculate_effective_acceleration(
    original_kspace: np.ndarray,
    undersampled_kspace: np.ndarray,
) -> float:
    """
    Calculate acceleration relative to the measured support of
    the original Calgary k-space.
    """

    maximum = float(
        np.max(np.abs(original_kspace))
    )

    if maximum == 0:
        raise ValueError("Original k-space is empty.")

    threshold = maximum * 1e-10

    original_support = np.any(
        np.abs(original_kspace) > threshold,
        axis=0,
    )

    undersampled_support = np.any(
        np.abs(undersampled_kspace) > threshold,
        axis=0,
    )

    original_samples = np.count_nonzero(
        original_support
    )

    undersampled_samples = np.count_nonzero(
        undersampled_support
    )

    if undersampled_samples == 0:
        raise ValueError(
            "The undersampled k-space contains no samples."
        )

    return original_samples / undersampled_samples


# ============================================================
# RSS reconstruction
# ============================================================

def reconstruct_rss(
    kspace: np.ndarray,
) -> np.ndarray:
    """
    Reconstruct a multi-coil image using centered 3D inverse FFT
    followed by root-sum-of-squares coil combination.

    Input:
        (coil, ky, kz, kx)

    Output:
        (ky, kz, x)
    """

    coil_images = ifftc(
        kspace,
        axes=(1, 2, 3),
    )

    # Calgary-specific spatial centering.
    coil_images = np.fft.ifftshift(
        coil_images,
        axes=(1, 2),
    )

    rss = np.sqrt(
        np.sum(
            np.abs(coil_images) ** 2,
            axis=0,
        )
    )

    return rss.astype(np.float32)


# ============================================================
# BART ESPIRiT/PICS reconstruction
# ============================================================

def reconstruct_espirit_pics(
    kspace: np.ndarray,
    calibration_size: int,
    regularization: float,
    bart,
    use_l1: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimate ESPIRiT sensitivity maps and reconstruct using PICS.

    Input shape:
        (coil, ky, kz, kx)

    BART input shape:
        (ky, kz, kx, coil)

    Returns:
        magnitude reconstruction: (ky, kz, x)
        sensitivity maps
    """

    # BART dimensions:
    #   0 = readout/spatial encoding 1
    #   1 = phase encoding 1
    #   2 = phase encoding 2
    #   3 = receiver coils
    bart_kspace = np.transpose(
        kspace,
        (1, 2, 3, 0),
    ).astype(np.complex64)

    print("BART k-space shape:", bart_kspace.shape)

    print(
        "Estimating ESPIRiT sensitivity maps "
        f"with calibration size {calibration_size}..."
    )

    sensitivity_maps = bart(
        1,
        f"ecalib -m 1 -r {calibration_size}",
        bart_kspace,
    )

    print(
        "Sensitivity-map shape:",
        sensitivity_maps.shape,
    )

    # Build dynamic BART command based on PI vs CS methodology
    cmd = f"pics -S -l1 -r {regularization}" if use_l1 else f"pics -S -r {regularization}"

    print(
        f"Running PICS reconstruction with command: {cmd}"
    )

    reconstruction = bart(
        1,
        cmd,
        bart_kspace,
        sensitivity_maps,
    )

    reconstruction = np.squeeze(reconstruction)

    if reconstruction.ndim != 3:
        raise ValueError(
            "Expected a 3D PICS reconstruction after squeezing, "
            f"but received shape {reconstruction.shape}."
        )

    magnitude = np.abs(
        reconstruction
    ).astype(np.float32)

    # Match the Calgary spatial centering used in the RSS volume.
    magnitude = np.fft.ifftshift(
        magnitude,
        axes=(0, 1),
    )

    print(
        "PICS reconstruction shape:",
        magnitude.shape,
    )

    return magnitude, sensitivity_maps


# ============================================================
# Visualization
# ============================================================

def normalize_for_display(
    image: np.ndarray,
    percentile: float = 99.5,
) -> tuple[float, float]:
    """Return robust display limits."""

    positive_values = image[
        np.isfinite(image)
    ]

    if positive_values.size == 0:
        return 0.0, 1.0

    vmax = float(
        np.percentile(
            positive_values,
            percentile,
        )
    )

    if vmax <= 0:
        vmax = 1.0

    return 0.0, vmax


def save_three_way_comparison(
    full_rss: np.ndarray,
    zero_filled: np.ndarray,
    pics_reconstruction: np.ndarray,
    output_path: Path,
    acceleration: int,
    slice_index: int | None = None,
    slice_axis: int = 2,
) -> None:
    """
    Display and save the fully sampled, zero-filled, and PICS
    reconstructions.
    """

    if full_rss.shape != zero_filled.shape:
        raise ValueError(
            "Full RSS and zero-filled volumes have different shapes: "
            f"{full_rss.shape} vs {zero_filled.shape}"
        )

    if full_rss.shape != pics_reconstruction.shape:
        raise ValueError(
            "Full RSS and PICS volumes have different shapes: "
            f"{full_rss.shape} vs {pics_reconstruction.shape}"
        )

    if slice_index is None:
        slice_index = full_rss.shape[slice_axis] // 2

    if not 0 <= slice_index < full_rss.shape[slice_axis]:
        raise IndexError(
            f"slice_index={slice_index} is outside "
            f"0:{full_rss.shape[slice_axis] - 1}"
        )

    if slice_axis == 0:
        full_slice = full_rss[slice_index, :, :].T
        zero_slice = zero_filled[slice_index, :, :].T
        pics_slice = pics_reconstruction[slice_index, :, :].T
    elif slice_axis == 1:
        full_slice = full_rss[:, slice_index, :].T
        zero_slice = zero_filled[:, slice_index, :].T
        pics_slice = pics_reconstruction[:, slice_index, :].T
    else:
        full_slice = full_rss[:, :, slice_index].T
        zero_slice = zero_filled[:, :, slice_index].T
        pics_slice = pics_reconstruction[:, :, slice_index].T

    diff_zero = np.abs(full_slice - zero_slice)
    diff_pics = np.abs(full_slice - pics_slice)

    vmin, vmax = normalize_for_display(
        full_slice
    )
    
    diff_vmax = vmax * 0.15

    figure, axes = plt.subplots(
        2,
        3,
        figsize=(18, 12),
    )

    images = [
        full_slice,
        zero_slice,
        pics_slice,
        None,
        diff_zero,
        diff_pics,
    ]

    titles = [
        "Fully sampled RSS",
        f"Zero-filled R={acceleration}",
        f"ESPIRiT/PICS R={acceleration}",
        "",
        "Difference (x6.6)",
        "Difference (x6.6)",
    ]

    for i, (image, title) in enumerate(zip(images, titles)):
        row = i // 3
        col = i % 3
        axis = axes[row, col]
        
        if image is None:
            axis.axis("off")
            continue

        img_vmax = vmax if row == 0 else diff_vmax
        cmap = "gray" if row == 0 else "viridis"

        axis.imshow(
            image,
            cmap=cmap,
            origin="upper",
            vmin=0,
            vmax=img_vmax,
        )

        axis.set_title(title)
        axis.axis("off")

    plane_names = {0: "Coronal", 1: "Sagittal", 2: "Axial"}
    plane_name = plane_names.get(slice_axis, "Unknown")

    figure.suptitle(
        f"{plane_name} slice {slice_index}",
        fontsize=15,
    )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    # plt.show()
    plt.close(figure)


def save_mask_figure(
    mask: np.ndarray,
    output_path: Path,
    acceleration: int,
) -> None:
    """
    Save the ky sampling pattern.

    The mask is constant along kz and kx, so one profile is shown.
    """

    ky_profile = mask[
        0,
        :,
        mask.shape[2] // 2,
        mask.shape[3] // 2,
    ]

    figure, axis = plt.subplots(
        figsize=(12, 3),
    )

    axis.imshow(
        ky_profile[np.newaxis, :],
        cmap="gray",
        aspect="auto",
        interpolation="nearest",
    )

    axis.set_title(
        f"Regular Cartesian R={acceleration} sampling with central ACS lines"
    )

    axis.set_xlabel("ky index")
    axis.set_yticks([])

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)


# ============================================================
# Main pipeline
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create a retrospective R=2 acquisition from fully "
            "sampled Calgary-Campinas k-space and compare RSS, "
            "zero-filled, and ESPIRiT/PICS reconstructions."
        )
    )

    parser.add_argument(
        "input_file",
        type=Path,
        help="Path to one fully sampled Calgary HDF5 file.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results_multiR"),
        help="Output directory.",
    )

    parser.add_argument(
        "--acs-lines",
        type=int,
        default=24,
        help=(
            "Number of fully sampled central ky lines. "
            "Default: 24."
        ),
    )

    parser.add_argument(
        "--calibration-size",
        type=int,
        default=24,
        help=(
            "ESPIRiT calibration-region size passed to "
            "BART ecalib. Default: 24."
        ),
    )

    parser.add_argument(
        "--regularization",
        type=float,
        default=0.001,
        help=(
            "PICS L2 regularization strength. "
            "Default: 0.001."
        ),
    )

    parser.add_argument(
        "--offset",
        type=int,
        choices=(0, 1),
        default=0,
        help="Keep even or odd ky lines. Default: 0.",
    )

    parser.add_argument(
        "--slice-index",
        type=int,
        default=None,
        help=(
            "Slice to display along the final volume axis. "
            "Default: central slice."
        ),
    )

    parser.add_argument(
        "--keep-partial-fourier-tail",
        action="store_true",
        help=(
            "Do not explicitly zero the nominal unacquired "
            "Calgary kz tail."
        ),
    )

    args = parser.parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # 1. Load the Calgary hybrid-space data.
    # --------------------------------------------------------

    hybrid = load_calgary_hybrid(
        args.input_file
    )

    # --------------------------------------------------------
    # 2. Remove nominally unacquired partial-Fourier samples.
    # --------------------------------------------------------

    if not args.keep_partial_fourier_tail:
        hybrid = zero_partial_fourier_tail(
            hybrid
        )

    # --------------------------------------------------------
    # 3. Recover full 3D multi-coil k-space.
    # --------------------------------------------------------

    full_kspace = hybrid_to_full_kspace(
        hybrid
    )

    print(
        "Full k-space shape:",
        full_kspace.shape,
    )

    number_of_coils, ky_size, kz_size, kx_size = (
        full_kspace.shape
    )

    print("Number of coils:", number_of_coils)
    print("ky size:", ky_size)
    print("kz size:", kz_size)
    print("kx size:", kx_size)

    # --------------------------------------------------------
    # 6. Fully sampled RSS reference.
    # --------------------------------------------------------

    print("Reconstructing fully sampled RSS reference...")
    full_rss = reconstruct_rss(full_kspace)

    bart = import_bart()

    print("Checking BART version...")
    bart(0, "version")

    np.save(
        args.output_dir / "rss_fully_sampled.npy",
        full_rss,
    )

    for acceleration in [2, 6]:
        print(f"\n{'='*50}")
        print(f"Running reconstruction for R={acceleration}")
        print(f"{'='*50}\n")

        mask = make_regular_mask(
            ky_size=ky_size,
            kz_size=kz_size,
            kx_size=kx_size,
            acs_lines=args.acs_lines,
            offset=args.offset,
            acceleration=acceleration,
        )

        print("Mask shape:", mask.shape)

        r_kspace = (full_kspace * mask).astype(np.complex64)

        effective_acceleration = calculate_effective_acceleration(
            full_kspace,
            r_kspace,
        )

        print(
            "Effective acceleration relative to acquired support:",
            f"{effective_acceleration:.4f}",
        )

        print(f"Reconstructing zero-filled R={acceleration} RSS volume...")
        zero_filled_rss = reconstruct_rss(r_kspace)

        print(f"\n--- Reconstructing with PICS (R={acceleration}) ---")
        use_l1 = (acceleration == 6)

        pics_reconstruction, sensitivity_maps = reconstruct_espirit_pics(
            kspace=r_kspace,
            calibration_size=args.calibration_size,
            regularization=args.regularization,
            bart=bart,
            use_l1=use_l1,
        )

        # Save arrays
        np.save(args.output_dir / f"mask_R{acceleration}.npy", mask.squeeze(axis=0))
        np.save(args.output_dir / f"kspace_R{acceleration}.npy", r_kspace)
        np.save(args.output_dir / f"rss_R{acceleration}_zero_filled.npy", zero_filled_rss)
        np.save(args.output_dir / f"reconstruction_R{acceleration}_espirit_pics.npy", pics_reconstruction)
        np.save(args.output_dir / f"espirit_sensitivity_maps_R{acceleration}.npy", sensitivity_maps)

        # Save figures
        for axis_idx, plane_name in zip((2, 1, 0), ("axial", "sagittal", "coronal")):
            save_three_way_comparison(
                full_rss=full_rss,
                zero_filled=zero_filled_rss,
                pics_reconstruction=pics_reconstruction,
                output_path=(args.output_dir / f"comparison_R{acceleration}_three_methods_{plane_name}.png"),
                acceleration=acceleration,
                slice_index=args.slice_index if axis_idx == 2 else None,
                slice_axis=axis_idx,
            )

        save_mask_figure(
            mask=mask,
            output_path=(args.output_dir / f"sampling_mask_R{acceleration}.png"),
            acceleration=acceleration,
        )

    print()
    print("Finished successfully.")
    print(
        "Outputs saved to:",
        args.output_dir.resolve(),
    )


if __name__ == "__main__":
    main()

'''
import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np



def fftc(data: np.ndarray, axes: tuple[int, ...]) -> np.ndarray:
    """Centered orthonormal FFT."""
    return np.fft.fftshift(
        np.fft.fftn(
            np.fft.ifftshift(data, axes=axes),
            axes=axes,
            norm="ortho",
        ),
        axes=axes,
    )


def ifftc(data: np.ndarray, axes: tuple[int, ...]) -> np.ndarray:
    """Centered orthonormal inverse FFT."""
    return np.fft.fftshift(
        np.fft.ifftn(
            np.fft.ifftshift(data, axes=axes),
            axes=axes,
            norm="ortho",
        ),
        axes=axes,
    )


def load_calgary_hybrid(filename: str) -> np.ndarray:
    """
    Load Calgary-Campinas hybrid-space data.

    Stored shape:
        (x, ky, kz, 2 * coils)

    Output shape:
        (coils, ky, kz, x)
    """
    with h5py.File(filename, "r") as file:
        raw = file["kspace"][()]

    print(f"Raw shape: {raw.shape}")
    print(f"Raw dtype: {raw.dtype}")

    if raw.ndim != 4:
        raise ValueError(f"Expected a 4D array, received {raw.shape}")

    if raw.shape[-1] % 2 != 0:
        raise ValueError(
            "The final dimension must contain interleaved real/imaginary channels."
        )

    # Interleaved representation:
    # real coil 1, imaginary coil 1, real coil 2, imaginary coil 2, ...
    hybrid = (
        raw[..., 0::2].astype(np.float32)
        + 1j * raw[..., 1::2].astype(np.float32)
    )

    # (x, ky, kz, coil) -> (coil, ky, kz, x)
    hybrid = np.transpose(hybrid, (3, 1, 2, 0))

    print(f"Complex hybrid-space shape: {hybrid.shape}")

    return hybrid.astype(np.complex64)


def remove_partial_fourier_tail(hybrid: np.ndarray) -> np.ndarray:
    """
    Explicitly zero the known unacquired Calgary partial-Fourier tail.

    The input shape is:
        (coil, ky, kz, x)
    """
    hybrid = hybrid.copy()
    kz_size = hybrid.shape[2]

    acquired_kz = {
        170: 145,
        174: 148,
        180: 153,
    }

    if kz_size in acquired_kz:
        cutoff = acquired_kz[kz_size]
        hybrid[:, :, cutoff:, :] = 0
        print(f"Zeroed partial-Fourier tail: kz indices {cutoff}:{kz_size}")
    else:
        print(
            f"Warning: no predefined partial-Fourier cutoff for kz={kz_size}. "
            "The input was left unchanged."
        )

    return hybrid


def hybrid_to_kspace(hybrid: np.ndarray) -> np.ndarray:
    """
    Convert Calgary hybrid-space data to complete 3D k-space.

    The x-axis is stored in image space, so a centered FFT is applied
    along the final axis.

    Input/output shape:
        (coil, ky, kz, kx)
    """
    return fftc(hybrid, axes=(3,)).astype(np.complex64)


def make_regular_r2_mask(
    ky_size: int,
    kz_size: int,
    kx_size: int,
    offset: int = 0,
    acs_lines: int = 0,
) -> np.ndarray:
    """
    Create a regular Cartesian mask with nominal acceleration R=2.

    Undersampling is applied along ky only.

    Parameters
    ----------
    offset:
        0 keeps even ky lines.
        1 keeps odd ky lines.

    acs_lines:
        Number of fully sampled central ky lines.
        Use 0 for an exact regular R=2 mask.
        Use a positive value, such as 24, for a GRAPPA-like mask.
    """
    if offset not in (0, 1):
        raise ValueError("offset must be either 0 or 1")

    ky_mask = np.zeros(ky_size, dtype=np.float32)
    ky_mask[offset::2] = 1.0

    if acs_lines > 0:
        acs_lines = min(acs_lines, ky_size)

        center = ky_size // 2
        start = center - acs_lines // 2
        end = start + acs_lines

        ky_mask[start:end] = 1.0

    # Broadcastable shape for:
    # (coil, ky, kz, kx)
    mask = ky_mask[None, :, None, None]

    # Expand for saving and visualization.
    mask_3d = np.broadcast_to(
        mask,
        (1, ky_size, kz_size, kx_size),
    ).copy()

    return mask_3d


def reconstruct_rss(kspace: np.ndarray) -> np.ndarray:
    """
    Reconstruct coil images using a centered 3D inverse FFT,
    followed by RSS coil combination.

    Input:
        (coil, ky, kz, kx)

    Output:
        (ky, kz, x)
    """
    coil_images = ifftc(
        kspace,
        axes=(1, 2, 3),
    )

    # Dataset-specific spatial centering used for Calgary.
    coil_images = np.fft.ifftshift(
        coil_images,
        axes=(1, 2),
    )

    rss = np.sqrt(
        np.sum(np.abs(coil_images) ** 2, axis=0)
    )

    return rss.astype(np.float32)


def calculate_effective_acceleration(
    full_kspace: np.ndarray,
    undersampled_kspace: np.ndarray,
) -> float:
    """
    Calculate acceleration relative to the actually acquired support
    of the original Calgary scan.
    """
    scale = np.max(np.abs(full_kspace))
    threshold = scale * 1e-10

    full_support = np.any(
        np.abs(full_kspace) > threshold,
        axis=0,
    )

    undersampled_support = np.any(
        np.abs(undersampled_kspace) > threshold,
        axis=0,
    )

    n_full = np.count_nonzero(full_support)
    n_under = np.count_nonzero(undersampled_support)

    if n_under == 0:
        raise ValueError("The undersampled k-space contains no acquired samples.")

    return n_full / n_under


def save_central_slices(
    full_rss: np.ndarray,
    r2_rss: np.ndarray,
    output_path: Path,
) -> None:
    """Save a comparison of central slices."""
    slice_index = full_rss.shape[2] // 2

    full_slice = full_rss[:, :, slice_index].T
    r2_slice = r2_rss[:, :, slice_index].T

    vmax = np.percentile(full_slice, 99.5)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(
        full_slice,
        cmap="gray",
        origin="lower",
        vmin=0,
        vmax=vmax,
    )
    plt.title("Fully sampled RSS")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(
        r2_slice,
        cmap="gray",
        origin="lower",
        vmin=0,
        vmax=vmax,
    )
    plt.title("Zero-filled reconstruction, nominal R=2")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Retrospectively undersample fully sampled Calgary-Campinas "
            "multi-coil data and reconstruct an RSS volume."
        )
    )

    parser.add_argument(
        "input_file",
        type=str,
        help="Path to the fully sampled Calgary HDF5 file.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="reconstruction_output",
        help="Directory in which to save the outputs.",
    )

    parser.add_argument(
        "--acs-lines",
        type=int,
        default=0,
        help=(
            "Number of fully sampled central ky lines. "
            "Use 0 for exact regular R=2 undersampling."
        ),
    )

    parser.add_argument(
        "--offset",
        type=int,
        choices=[0, 1],
        default=0,
        help="Keep either even or odd ky lines.",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load hybrid-space multi-coil data.
    hybrid = load_calgary_hybrid(args.input_file)

    # 2. Remove nominally unacquired partial-Fourier tail.
    hybrid = remove_partial_fourier_tail(hybrid)

    # 3. Recover complete 3D k-space.
    full_kspace = hybrid_to_kspace(hybrid)

    print(f"Full k-space shape: {full_kspace.shape}")

    _, ky_size, kz_size, kx_size = full_kspace.shape

    # 4. Generate nominal R=2 regular Cartesian mask.
    mask = make_regular_r2_mask(
        ky_size=ky_size,
        kz_size=kz_size,
        kx_size=kx_size,
        offset=args.offset,
        acs_lines=args.acs_lines,
    )

    # 5. Retrospective undersampling.
    r2_kspace = full_kspace * mask

    acceleration = calculate_effective_acceleration(
        full_kspace,
        r2_kspace,
    )

    print(f"Effective acceleration relative to acquired support: {acceleration:.3f}")

    # 6. Reconstruct both the reference and R=2 volumes.
    full_rss = reconstruct_rss(full_kspace)
    r2_rss = reconstruct_rss(r2_kspace)

    print(f"Full RSS shape: {full_rss.shape}")
    print(f"R=2 RSS shape: {r2_rss.shape}")

    # 7. Save results.
    np.save(output_dir / "mask_R2.npy", mask.squeeze(0))
    np.save(output_dir / "rss_fully_sampled.npy", full_rss)
    np.save(output_dir / "rss_R2_zero_filled.npy", r2_rss)

    save_central_slices(
        full_rss=full_rss,
        r2_rss=r2_rss,
        output_path=output_dir / "comparison_R2.png",
    )

    print(f"Saved outputs to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()

'''