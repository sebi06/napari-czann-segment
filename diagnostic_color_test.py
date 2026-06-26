#!/usr/bin/env python
"""
Simple diagnostic test to check:
1. What format the image is in when loaded by napari/czitools
2. Whether color conversion affects inference results
3. Which direction (if any) produces correct results
"""

import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# Try to load the test image using bioio (which napari uses)
def load_test_image():
    """Load test image in the same way napari would"""
    try:
        from bioio import BioImage

        image_path = Path("src/napari_czann_segment/_data/Image 12 small.czi")
        if not image_path.exists():
            logger.error(f"Image not found: {image_path}")
            return None

        img = BioImage(str(image_path))
        img_array = np.asarray(img.data)
        logger.info(f"Image loaded successfully")
        logger.info(f"  Full shape: {img_array.shape}")
        logger.info(f"  Dtype: {img_array.dtype}")
        logger.info(f"  Dims: {img.dims if hasattr(img, 'dims') else 'N/A'}")

        return img_array
    except Exception as e:
        logger.error(f"Failed to load image: {e}")
        return None


def analyze_color_channels(img_array):
    """Analyze the color channels to understand the image format"""
    logger.info("\n=== COLOR CHANNEL ANALYSIS ===")

    # Squeeze to get to the image data
    img_squeezed = np.squeeze(img_array)
    logger.info(f"After squeeze: {img_squeezed.shape}")

    if img_squeezed.ndim >= 3:
        # Check if last dimension is color channels
        last_dim = img_squeezed.shape[-1]
        if 1 <= last_dim <= 4:
            logger.info(f"\nLast dimension is {last_dim} (likely color channels)")

            # Get statistics for each channel
            if last_dim >= 3:
                logger.info("\nChannel statistics (first 1000x1000 pixels):")
                test_region = img_squeezed[:1000, :1000, :]
                for c in range(min(3, last_dim)):
                    channel_data = test_region[..., c]
                    logger.info(
                        f"  Channel {c}: min={channel_data.min()}, max={channel_data.max()}, "
                        f"mean={channel_data.mean():.1f}, median={np.median(channel_data):.1f}"
                    )

                # Check if channels are similar (likely grayscale) or different (likely RGB)
                if last_dim >= 3:
                    ch0_mean = test_region[..., 0].mean()
                    ch1_mean = test_region[..., 1].mean()
                    ch2_mean = test_region[..., 2].mean()
                    means = [ch0_mean, ch1_mean, ch2_mean]
                    max_diff = max(means) - min(means)

                    logger.info(f"\nChannel mean comparison:")
                    logger.info(f"  Ch0 mean: {ch0_mean:.1f}")
                    logger.info(f"  Ch1 mean: {ch1_mean:.1f}")
                    logger.info(f"  Ch2 mean: {ch2_mean:.1f}")
                    logger.info(f"  Max difference: {max_diff:.1f}")

                    if max_diff < 10:
                        logger.warning("  → Channels are very similar (likely GRAYSCALE or near-grayscale)")
                    else:
                        logger.info("  → Channels are different (likely RGB/BGR color image)")


def test_inference_with_directions():
    """Test inference with both conversion directions"""
    logger.info("\n=== TESTING INFERENCE WITH DIFFERENT CONVERSION DIRECTIONS ===")

    from src.napari_czann_segment.predict import predict_ndarray
    from src.napari_czann_segment.get_testdata import get_modelfile

    img_array = load_test_image()
    if img_array is None:
        return

    model_file = get_modelfile("260513_2025285_NMI-D_256_Uincep3_v1.czseg")
    logger.info(f"\nUsing model: {Path(model_file).name}")
    logger.info(f"Input image shape: {img_array.shape}")

    # Squeeze to 2D for testing (take first Z, T slice)
    img_test = np.squeeze(img_array)[..., :256, :256, :]  # Take small region for speed
    logger.info(f"Test region shape: {img_test.shape}")

    try:
        # Test 1: WITHOUT conversion
        logger.info("\n--- Test 1: WITHOUT conversion (convert_rgb_to_bgr=False) ---")
        result_no_conv = predict_ndarray(
            czann_file=str(model_file),
            img=img_test,
            border="auto",
            use_gpu=False,
            do_rescale=True,
            convert_rgb_to_bgr=False,
        )
        _, seg_no_conv = result_no_conv
        logger.info(f"Result shape: {seg_no_conv.shape}")
        logger.info(f"Unique values: {np.unique(seg_no_conv)}")
        logger.info(f"Class distribution: {np.bincount(seg_no_conv.astype(int).flatten())}")

        # Test 2: WITH conversion (current direction: RGB→BGR)
        logger.info("\n--- Test 2: WITH conversion (convert_rgb_to_bgr=True, RGB→BGR direction) ---")
        result_with_conv = predict_ndarray(
            czann_file=str(model_file),
            img=img_test,
            border="auto",
            use_gpu=False,
            do_rescale=True,
            convert_rgb_to_bgr=True,
        )
        _, seg_with_conv = result_with_conv
        logger.info(f"Result shape: {seg_with_conv.shape}")
        logger.info(f"Unique values: {np.unique(seg_with_conv)}")
        logger.info(f"Class distribution: {np.bincount(seg_with_conv.astype(int).flatten())}")

        # Compare
        are_identical = np.array_equal(seg_no_conv, seg_with_conv)
        logger.info(f"\n--- COMPARISON ---")
        logger.info(f"Results are identical: {are_identical}")

        if not are_identical:
            diff_pixels = np.sum(seg_no_conv != seg_with_conv)
            logger.info(f"Pixels that differ: {diff_pixels} / {seg_no_conv.size}")
        else:
            logger.warning("Conversion had NO EFFECT on results - either:")
            logger.warning("  1. Image doesn't have 3 channels (unlikely - we verified it does)")
            logger.warning("  2. Conversion is happening but doesn't change results (wrong direction?)")
            logger.warning("  3. Color order is not the root cause of the problem")

    except Exception as e:
        logger.error(f"Error during inference: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("DIAGNOSTIC TEST: Color Channel Analysis & Conversion Testing")
    logger.info("=" * 60)

    img_array = load_test_image()
    if img_array is not None:
        analyze_color_channels(img_array)
        test_inference_with_directions()

    logger.info("\n" + "=" * 60)
    logger.info("Diagnostic test complete")
    logger.info("=" * 60)
