# -*- coding: utf-8 -*-

"""
Tests for CZSEG parser functionality
"""

import pytest
import tempfile
from pathlib import Path
from napari_czann_segment.czseg_parser import extract_czseg_model, parse_czseg_xml
from napari_czann_segment.get_testdata import get_modelfile
from czmodel import ModelType


def test_extract_czseg_with_tile_dimensions():
    """Test CZSEG extraction with complete XML (includes TotalTileHeight/Width)."""
    czseg_file = get_modelfile(name_czann="PGC_20X_nucleus_detector.czseg")

    with tempfile.TemporaryDirectory() as temp_dir:
        metadata, model_path, expects_bgr, class_colors = extract_czseg_model(czseg_file, Path(temp_dir))

        # Verify metadata
        assert metadata.model_id == "cd45c952-27d0-4f0f-888a-cf560ee5728f"
        assert metadata.model_name == "PGC_CD7_20XNA0.95"
        assert metadata.model_type == ModelType.SINGLE_CLASS_SEMANTIC_SEGMENTATION
        assert metadata.input_shape == [1024, 1024, 1]
        assert metadata.output_shape == [1024, 1024, 2]
        assert metadata.min_overlap == [128, 128]
        assert metadata.classes == ["background", "nuc"]
        assert metadata.scaling is True

        # Grayscale model must not request BGR channel ordering
        assert expects_bgr is False

        # Class colors are parsed from the XML (one RGB tuple per class)
        assert class_colors == [(255, 0, 0), (0, 255, 0)]

        # Verify model file exists
        assert model_path.exists()
        assert model_path.suffix == ".model"


def test_czseg_vs_czann_compatibility():
    """Test that CZSEG and CZANN versions of same model produce compatible metadata."""
    from czmodel.core.util._extract_model import extract_czann_model

    czseg_file = get_modelfile(name_czann="PGC_20X_nucleus_detector.czseg")
    czann_file = get_modelfile(name_czann="PGC_20X_nucleus_detector.czann")

    with tempfile.TemporaryDirectory() as temp_dir1:
        with tempfile.TemporaryDirectory() as temp_dir2:
            czseg_metadata, _, _, _ = extract_czseg_model(czseg_file, Path(temp_dir1))
            czann_metadata, _ = extract_czann_model(czann_file, Path(temp_dir2))

            # Key fields should match
            assert czseg_metadata.model_id == czann_metadata.model_id
            assert czseg_metadata.model_type == czann_metadata.model_type
            assert czseg_metadata.input_shape == czann_metadata.input_shape
            assert czseg_metadata.output_shape == czann_metadata.output_shape
            assert czseg_metadata.min_overlap == czann_metadata.min_overlap
            assert czseg_metadata.classes == czann_metadata.classes


def test_invalid_czseg_file():
    """Test that invalid CZSEG files raise appropriate errors."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Test non-existent file
        with pytest.raises(FileNotFoundError):
            extract_czseg_model(Path("nonexistent.czseg"), Path(temp_dir))

        # Test wrong extension
        wrong_ext = Path(temp_dir) / "test.txt"
        wrong_ext.touch()
        with pytest.raises(ValueError, match="must have .czseg extension"):
            extract_czseg_model(wrong_ext, Path(temp_dir))
