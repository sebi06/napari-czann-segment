# -*- coding: utf-8 -*-

import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from czmodel.core.util._extract_model import extract_czann_model

from napari_czann_segment import get_testdata
from napari_czann_segment.onnx_inference import (
    ONNXRUNTIME_AVAILABLE,
    ManagedOnnxSession,
    OnnxInferencer,
    is_gpu_available,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture()
def seg_model_path():
    """Extract the simple segmentation model to a temp dir and yield the ONNX path."""
    czann = get_testdata.get_modelfile("simple_nuclei_segmodel.czann")
    with tempfile.TemporaryDirectory() as td:
        _, model_path = extract_czann_model(path=czann, target_dir=Path(td))
        yield str(model_path)


@pytest.fixture()
def reg_model_path():
    """Extract the simple regression model to a temp dir and yield the ONNX path."""
    czann = get_testdata.get_modelfile("simple_regmodel.czann")
    with tempfile.TemporaryDirectory() as td:
        _, model_path = extract_czann_model(path=czann, target_dir=Path(td))
        yield str(model_path)


# ---------------------------------------------------------------------------
# is_gpu_available
# ---------------------------------------------------------------------------


class TestIsGpuAvailable:
    def test_returns_bool(self):
        result = is_gpu_available()
        assert isinstance(result, bool)

    def test_returns_false_when_onnxruntime_missing(self):
        with patch("napari_czann_segment.onnx_inference.ONNXRUNTIME_AVAILABLE", False):
            assert is_gpu_available() is False


# ---------------------------------------------------------------------------
# ManagedOnnxSession
# ---------------------------------------------------------------------------


class TestManagedOnnxSession:
    def test_cpu_session(self, seg_model_path):
        with ManagedOnnxSession(seg_model_path, providers=["CPUExecutionProvider"]) as sess:
            assert sess is not None
            assert sess.get_inputs() is not None

    def test_fallback_to_cpu_on_bad_provider(self, seg_model_path):
        """If a bogus CUDA-like provider is requested, session falls back to CPU."""
        bad_providers = [
            ("CUDAExecutionProvider", {"device_id": "999"}),
            "CPUExecutionProvider",
        ]
        with ManagedOnnxSession(seg_model_path, providers=bad_providers) as sess:
            # Should have fallen back to CPU without raising
            assert sess is not None

    def test_raises_when_onnxruntime_unavailable(self, seg_model_path):
        with patch("napari_czann_segment.onnx_inference.ONNXRUNTIME_AVAILABLE", False):
            with pytest.raises(ImportError, match="onnxruntime is not available"):
                ManagedOnnxSession(seg_model_path, providers=["CPUExecutionProvider"])


# ---------------------------------------------------------------------------
# OnnxInferencer – shape introspection
# ---------------------------------------------------------------------------


class TestOnnxInferencerShapes:
    def test_get_input_shape(self, seg_model_path):
        inf = OnnxInferencer(seg_model_path)
        shape = inf.get_input_shape()
        assert len(shape) == 4
        # Batch dim is typically None, spatial dims are ints
        assert all(isinstance(d, (int, type(None))) for d in shape)

    def test_get_output_shape(self, seg_model_path):
        inf = OnnxInferencer(seg_model_path)
        shape = inf.get_output_shape()
        assert len(shape) == 4
        assert all(isinstance(d, (int, type(None))) for d in shape)

    def test_input_output_spatial_dims_consistent(self, reg_model_path):
        """For a regression model the spatial dims of input and output should match."""
        inf = OnnxInferencer(reg_model_path)
        in_shape = inf.get_input_shape()
        out_shape = inf.get_output_shape()
        # Spatial height/width (indices 2,3) should be equal
        assert in_shape[2] == out_shape[2]
        assert in_shape[3] == out_shape[3]


# ---------------------------------------------------------------------------
# OnnxInferencer – prediction on CPU
# ---------------------------------------------------------------------------


class TestOnnxInferencerPredict:
    def test_predict_single_tile(self, seg_model_path):
        inf = OnnxInferencer(seg_model_path)
        # Input shape is (batch, H, W, C)
        in_shape = inf.get_input_shape()
        h = in_shape[1] if in_shape[1] is not None else 64
        w = in_shape[2] if in_shape[2] is not None else 64
        c = in_shape[3] if in_shape[3] is not None else 1
        tile = np.random.rand(h, w, c).astype(np.float32)
        results = inf.predict([tile], use_gpu=False)
        assert len(results) == 1
        assert isinstance(results[0], np.ndarray)

    def test_predict_batch(self, reg_model_path):
        inf = OnnxInferencer(reg_model_path)
        # Input shape is (batch, H, W, C)
        in_shape = inf.get_input_shape()
        h = in_shape[1] if in_shape[1] is not None else 64
        w = in_shape[2] if in_shape[2] is not None else 64
        c = in_shape[3] if in_shape[3] is not None else 1
        tiles = [np.random.rand(h, w, c).astype(np.float32) for _ in range(3)]
        results = inf.predict(tiles, use_gpu=False)
        assert len(results) == 3
