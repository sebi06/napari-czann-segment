"""
Debug script to test tiling behavior with multi-channel images.
Run from napari-czann-segment root directory.
"""

import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.DEBUG, format="%(name)s - %(levelname)s - %(message)s")

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from napari_czann_segment import get_testdata
from napari_czann_segment.predict import predict_ndarray
from napari_czann_segment.utils import TileMethod


def test_single_channel():
    """Test with PGC model (grayscale) - known to work"""
    print("\n" + "=" * 80)
    print("TEST 1: PGC Model (Grayscale/Single-Channel)")
    print("=" * 80)

    model_file = get_testdata.get_modelfile("PGC_10X_nucleus_detector.czann")
    image_file = get_testdata.get_imagefile("Image 12 small.czi")

    print(f"Model: {model_file}")
    print(f"Image: {image_file}")

    # TODO: Load image and run predict_ndarray
    # This will show us the tile count for grayscale


def test_multi_channel():
    """Test with UIncep3 model (3-channel RGB) - producing wrong results"""
    print("\n" + "=" * 80)
    print("TEST 2: UIncep3 Model (3-Channel RGB)")
    print("=" * 80)

    model_file = get_testdata.get_modelfile("260513_2025285_NMI-D_256_Uincep3_v1.czann")
    image_file = get_testdata.get_imagefile("Image 12 small.czi")

    print(f"Model: {model_file}")
    print(f"Image: {image_file}")

    # TODO: Load image and run predict_ndarray
    # This will show us the tile count for 3-channel (should show the 18x increase)


if __name__ == "__main__":
    print("\nTiling Debug Script for napari-czann-segment")
    print("This script tests tiling behavior with different image types")

    test_single_channel()
    test_multi_channel()

    print("\nNote: Full implementation requires image loading from CZI format.")
    print("The logging output above will show CZTILE tile count and ROI details.")
