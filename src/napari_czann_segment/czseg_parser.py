# -*- coding: utf-8 -*-

#################################################################
# File        : czseg_parser.py
# Author      : GitHub Copilot
#
# Description : Parser for CZSEG model files (XML-based format)
#
#################################################################

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Tuple, Optional
import zipfile
import tempfile
from czmodel import ModelMetadata, ModelType
import onnxruntime as ort
import logging

logger = logging.getLogger(__name__)


def _infer_tile_size_from_onnx(model_path: Path) -> Tuple[int, int]:
    """
    Infer tile size (height, width) from ONNX model input shape.

    Parameters
    ----------
    model_path : Path
        Path to the ONNX model file.

    Returns
    -------
    Tuple[int, int]
        Tuple of (height, width) inferred from the model's input shape.

    Raises
    ------
    ValueError
        If the model input shape cannot be determined or is invalid.
    """
    try:
        # Create ONNX session to read model metadata
        session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])

        # Get input shape from the first input
        input_shape = session.get_inputs()[0].shape

        # Input shape is typically [batch, channels, height, width] or [batch, height, width, channels]
        # We need to identify which dimensions are height and width

        # Filter out batch dimension (usually -1, 1, or None) and channel dimension
        spatial_dims = [d for d in input_shape if isinstance(d, int) and d > 1]

        if len(spatial_dims) >= 2:
            # Assume the last two spatial dimensions are height and width
            # or if in NCHW format, dimensions 2 and 3
            if len(input_shape) == 4:  # Typical 4D tensor
                # Could be NCHW or NHWC
                # NCHW: [batch, channels, height, width]
                # NHWC: [batch, height, width, channels]
                if isinstance(input_shape[1], int) and input_shape[1] <= 10:
                    # Likely NCHW (small channel count)
                    height, width = input_shape[2], input_shape[3]
                else:
                    # Likely NHWC
                    height, width = input_shape[1], input_shape[2]
            else:
                # Fallback: use the last two dimensions
                height, width = spatial_dims[-2], spatial_dims[-1]

            # Ensure they're integers
            if isinstance(height, int) and isinstance(width, int):
                return height, width

        raise ValueError(f"Could not determine spatial dimensions from input shape: {input_shape}")

    except Exception as e:
        raise ValueError(f"Failed to infer tile size from ONNX model: {e}")


def parse_czseg_xml(xml_path: Path, model_path: Optional[Path] = None) -> dict:
    """
    Parse CZSEG XML file and extract model metadata.

    Parameters
    ----------
    xml_path : Path
        Path to the XML metadata file.
    model_path : Optional[Path]
        Path to the ONNX model file. If provided and tile dimensions are missing from XML,
        the parser will attempt to infer them from the ONNX model's input shape.

    Returns
    -------
    dict
        Dictionary containing parsed metadata fields.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Extract basic fields
    model_id = root.find("Id").text if root.find("Id") is not None else None
    model_name = root.find("ModelName").text if root.find("ModelName") is not None else None

    # Extract tile dimensions (may be missing in older CZSEG files)
    tile_height_elem = root.find("TotalTileHeight")
    tile_width_elem = root.find("TotalTileWidth")

    tile_height = int(tile_height_elem.text) if tile_height_elem is not None else None
    tile_width = int(tile_width_elem.text) if tile_width_elem is not None else None

    # If tile dimensions are missing, try to infer from ONNX model
    if (tile_height is None or tile_width is None) and model_path is not None:
        try:
            inferred_height, inferred_width = _infer_tile_size_from_onnx(model_path)
            if tile_height is None:
                tile_height = inferred_height
                logger.info(f"Inferred tile height from ONNX model: {tile_height}")
            if tile_width is None:
                tile_width = inferred_width
                logger.info(f"Inferred tile width from ONNX model: {tile_width}")
        except Exception as e:
            logger.warning(f"Could not infer tile size from ONNX model: {e}")

    # Final fallback to common default if still missing
    if tile_height is None:
        tile_height = 256
        logger.warning(f"TotalTileHeight missing in XML and could not be inferred. Using default: {tile_height}")
    if tile_width is None:
        tile_width = 256
        logger.warning(f"TotalTileWidth missing in XML and could not be inferred. Using default: {tile_width}")

    # Extract border size (in CZSEG this is per side, we need to double it for min_overlap)
    border_size = int(root.find("BorderSize").text) if root.find("BorderSize") is not None else 64
    min_overlap = [border_size * 2, border_size * 2]

    # Extract classes from TrainingClasses
    classes = []
    training_classes = root.find("TrainingClasses")
    if training_classes is not None:
        for item in training_classes.findall("Item"):
            class_name = item.get("Name")
            if class_name:
                classes.append(class_name)

    # Determine input/output shapes
    # Input: [height, width, channels]
    # Parse PixelType from the first <Channels><Item> to determine the actual channel count.
    # Counting <Item> elements would give 1 for Bgr24 (single item, 3 channels), which is wrong.
    _PIXEL_TYPE_CHANNELS = {
        "Bgr24": 3,
        "Rgb24": 3,
        "Bgra32": 4,
        "Rgba32": 4,
        "Gray8": 1,
        "Gray16": 1,
        "Gray32Float": 1,
    }
    channels_elem = root.find("Channels")
    num_input_channels = 1  # default (grayscale)
    if channels_elem is not None:
        items = channels_elem.findall("Item")
        if items:
            pixel_type = items[0].get("PixelType", "Gray8")
            num_input_channels = _PIXEL_TYPE_CHANNELS.get(pixel_type, 1)
            logger.info(f"Parsed PixelType='{pixel_type}' → {num_input_channels} input channel(s)")

    input_shape = [tile_height, tile_width, num_input_channels]

    # Output shape: [height, width, num_classes]
    # Number of output channels = number of classes
    num_output_channels = len(classes) if classes else 2
    output_shape = [tile_height, tile_width, num_output_channels]

    # Determine model type
    # Based on the presence of TrainingClasses, assume semantic segmentation
    model_type = ModelType.SINGLE_CLASS_SEMANTIC_SEGMENTATION

    # Check if scaling should be applied
    scale_inputs = root.find("ScaleInputsByBitdepth")
    scaling = scale_inputs.text.lower() == "true" if scale_inputs is not None else True

    return {
        "model_id": model_id,
        "model_name": model_name,
        "model_type": model_type,
        "input_shape": input_shape,
        "output_shape": output_shape,
        "min_overlap": min_overlap,
        "classes": classes,
        "scaling": scaling,
    }


def extract_czseg_model(path: Path | str, target_dir: Path | str) -> Tuple[ModelMetadata, Path]:
    """
    Extract CZSEG model file and parse metadata.

    This function mirrors the behavior of czmodel's extract_czann_model but for CZSEG files.

    Parameters
    ----------
    path : Path | str
        Path to the .czseg file.
    target_dir : Path | str
        Directory where the model contents will be extracted.

    Returns
    -------
    Tuple[ModelMetadata, Path]
        A tuple containing:
        - ModelMetadata: Parsed model metadata
        - Path: Path to the extracted ONNX model file

    Raises
    ------
    FileNotFoundError
        If the czseg file doesn't exist.
    ValueError
        If the czseg file doesn't contain expected files.
    """
    path = Path(path)
    target_dir = Path(target_dir)

    if not path.exists():
        raise FileNotFoundError(f"CZSEG file not found: {path}")

    if not path.suffix.lower() == ".czseg":
        raise ValueError(f"File must have .czseg extension, got: {path.suffix}")

    # Extract the zip file
    with zipfile.ZipFile(path, "r") as zip_ref:
        zip_ref.extractall(target_dir)

    # Find the XML and model files
    # CZSEG files contain {id}.xml and {id}.model
    xml_files = list(target_dir.glob("*.xml"))
    model_files = list(target_dir.glob("*.model"))

    if not xml_files:
        raise ValueError(f"No XML metadata file found in {path}")

    if not model_files:
        raise ValueError(f"No .model file found in {path}")

    xml_path = xml_files[0]
    model_path = model_files[0]

    # Parse the XML metadata, passing model_path for potential inference
    metadata_dict = parse_czseg_xml(xml_path, model_path=model_path)

    # Create ModelMetadata object
    model_metadata = ModelMetadata(
        model_type=metadata_dict["model_type"],
        input_shape=metadata_dict["input_shape"],
        output_shape=metadata_dict["output_shape"],
        model_id=metadata_dict["model_id"],
        min_overlap=metadata_dict["min_overlap"],
        classes=metadata_dict["classes"],
        model_name=metadata_dict["model_name"],
        scaling=metadata_dict["scaling"],
    )

    return model_metadata, model_path
