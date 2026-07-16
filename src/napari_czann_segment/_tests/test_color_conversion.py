#!/usr/bin/env python
"""
Diagnostic script to test RGB/BGR conversion issue.
This script manually tests inference with and without color conversion
to debug why the fix isn't working.

**Not a pytest test** — this runs full end-to-end inference twice on a
real CZI file and takes several minutes. Run manually with:
    python -m napari_czann_segment._tests.test_color_conversion

It is excluded from automatic pytest collection via the ``pytestmark``
below.
"""

import numpy as np
from pathlib import Path
import logging
import pytest

# Skip when collected by pytest; this is a manual diagnostic script.
pytestmark = pytest.mark.skip(reason="Manual diagnostic script — run directly, not via pytest")

# Setup logging
logging.basicConfig(level=logging.DEBUG, format="%(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

from napari_czann_segment.get_testdata import get_imagefile, get_modelfile
from napari_czann_segment.predict import predict_ndarray


def test_color_conversion():
    """Test the RGB/BGR conversion feature"""

    # Test 2: UIncep3 model (the problematic one)
    model_file = get_modelfile("260513_2025285_NMI-D_256_Uincep3_v1.czseg")
    image_file = get_imagefile("Image 12 small.czi")

    logger.info(f"Model file: {model_file}")
    logger.info(f"Image file: {image_file}")

    # Load image using bioio or similar
    try:
        from bioio import BioImage

        img = BioImage(str(image_file))
        # Get as numpy array - check what format it is
        img_array = np.asarray(img.data)
        logger.info(f"Image loaded with bioio: shape={img_array.shape}, dtype={img_array.dtype}")
        logger.info(f"Image channel info: {img.dims}, channels={img.channels if hasattr(img, 'channels') else 'N/A'}")
    except Exception as e:
        logger.warning(f"Could not load with bioio: {e}")
        # Try imageio as fallback
        try:
            import imageio

            img_array = imageio.imread(str(image_file))
            logger.info(f"Image loaded with imageio: shape={img_array.shape}, dtype={img_array.dtype}")
        except Exception as e2:
            logger.error(f"Could not load image: {e2}")
            return

    # Extract first 2D slice if 3D
    if img_array.ndim > 3:
        img_test = img_array[0]  # Take first Z slice
    elif img_array.ndim == 3:
        img_test = img_array
    else:
        img_test = img_array

    logger.info(f"Test image shape: {img_test.shape}, dtype: {img_test.dtype}")

    # Test WITHOUT color conversion
    logger.info("\n=== TEST 1: WITHOUT color conversion ===")
    try:
        result_no_conv = predict_ndarray(
            czann_file=str(model_file),
            img=img_test,
            border="auto",
            use_gpu=False,
            do_rescale=True,
            convert_rgb_to_bgr=False,
        )
        logger.info(f"Result shape: {result_no_conv[1].shape}, dtype: {result_no_conv[1].dtype}")
        logger.info(f"Result unique values: {np.unique(result_no_conv[1])}")
    except Exception as e:
        logger.error(f"Error without conversion: {e}")
        import traceback

        traceback.print_exc()
        return

    # Test WITH color conversion
    logger.info("\n=== TEST 2: WITH color conversion ===")
    try:
        result_with_conv = predict_ndarray(
            czann_file=str(model_file),
            img=img_test,
            border="auto",
            use_gpu=False,
            do_rescale=True,
            convert_rgb_to_bgr=True,
        )
        logger.info(f"Result shape: {result_with_conv[1].shape}, dtype: {result_with_conv[1].dtype}")
        logger.info(f"Result unique values: {np.unique(result_with_conv[1])}")
    except Exception as e:
        logger.error(f"Error with conversion: {e}")
        import traceback

        traceback.print_exc()
        return

    # Compare results
    logger.info("\n=== COMPARISON ===")
    if result_no_conv[1] is not None and result_with_conv[1] is not None:
        are_same = np.array_equal(result_no_conv[1], result_with_conv[1])
        logger.info(f"Results are identical (both ON and OFF same): {are_same}")

        if not are_same:
            diff_pixels = np.sum(result_no_conv[1] != result_with_conv[1])
            logger.info(f"Pixels that differ: {diff_pixels} / {result_no_conv[1].size}")
            logger.info(
                f"Difference stats - min: {(result_no_conv[1] - result_with_conv[1]).min()}, "
                f"max: {(result_no_conv[1] - result_with_conv[1]).max()}"
            )

    logger.info("\nDone!")


if __name__ == "__main__":
    test_color_conversion()
