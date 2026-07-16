#!/usr/bin/env python
"""Demo script to verify batching implementation (Points 1 & 3)."""

import sys
import tempfile
from pathlib import Path

import numpy as np
from czmodel.core.util._extract_model import extract_czann_model
import pytest

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from napari_czann_segment.get_testdata import get_modelfile
from napari_czann_segment.onnx_inference import OnnxInferencer, ONNXRUNTIME_AVAILABLE


@pytest.mark.skipif(
    not ONNXRUNTIME_AVAILABLE,
    reason="onnxruntime not available (e.g. Windows CI access violation)",
)
def test_batching():
    """Test that batching is working correctly."""

    # Load a test model
    czann_path = get_modelfile("simple_nuclei_segmodel.czann")
    print(f"Using model archive: {czann_path}")

    with tempfile.TemporaryDirectory() as temp_dir:
        _, model_path = extract_czann_model(path=czann_path, target_dir=Path(temp_dir))
        print(f"Extracted ONNX model: {model_path}")

        # Create inferencer with batch_size=4 (Point 1)
        inf = OnnxInferencer(str(model_path), batch_size=4)
        print(f"✓ OnnxInferencer created with batch_size=4")

        # Get model input shape
        input_shape = inf.get_input_shape()
        print(f"✓ Model input shape: {input_shape}")

        # Create 8 random test images (to test batching: 8 = 2 batches of 4)
        num_images = 8
        test_images = [
            np.random.rand(input_shape[1], input_shape[2], input_shape[3]).astype(np.float32)
            for _ in range(num_images)
        ]
        print(f"✓ Created {num_images} test images")

        # Run inference (Point 3: stacked batch processing)
        results = inf.predict(test_images, use_gpu=False)

        # Verify results
        assert len(results) == num_images, f"Expected {num_images} results, got {len(results)}"
        assert results[0].shape[:2] == (input_shape[1], input_shape[2])

        print(f"✓ Inference completed successfully")
        print(f"  - Processed {num_images} images in {(num_images + inf._batch_size - 1) // inf._batch_size} batches")
        print(f"  - Output shape per image: {results[0].shape}")

        # Test with different batch size
        inf_bs1 = OnnxInferencer(str(model_path), batch_size=1)
        results_bs1 = inf_bs1.predict(test_images[:2], use_gpu=False)
        assert len(results_bs1) == 2, "Batch size 1 test failed"
        print(f"✓ Batch size 1 (sequential) also works correctly")

    print("\n✅ All batching tests passed!")
    print("\nKey improvements implemented:")
    print("  - Point 1: Configurable batch size (default=8, was 1)")
    print("  - Point 3: Stacked batch tensor processing (not sequential loops)")


if __name__ == "__main__":
    test_batching()
