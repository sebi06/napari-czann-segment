# -*- coding: utf-8 -*-

#################################################################
# File        : test_czann_segment.py
# Author      : sebi06, Team Enchilada
#
# Disclaimer: This code is purely experimental. Feel free to
# use it at your own risk.
#
#################################################################

from napari_czann_segment import get_testdata
from napari_czann_segment.dock_widget import setup_log
from napari_czann_segment import predict, process_nd
from napari_czann_segment.onnx_inference import ONNXRUNTIME_AVAILABLE
from bioio import BioImage
from pathlib import Path
import tempfile
from napari_czann_segment.utils import TileMethod, SupportedWindow
from czmodel.pytorch.convert import DefaultConverter
import pytest
from typing import (
    Tuple,
)

logger = setup_log("Napari-CZANN")


@pytest.mark.parametrize(
    "czann, guid",
    [
        (
            "PGC_20X_nucleus_detector.czann",
            "cd45c952-27d0-4f0f-888a-cf560ee5728f",
        ),
        ("simple_regmodel_out5.czann", "b6f04bbd-2955-4996-8578-43d02d24f093"),
    ],
)
def test_extract_model(czann: str, guid: str) -> None:
    """
    Test the functionality of extracting a model using czann files.

    Parameters:
    - czann (str): The name of the czann file.
    - guid (str): The expected model ID.

    Returns:
    - None
    """

    # get data to test the functionality
    czann_file = get_testdata.get_modelfile(czann)

    with tempfile.TemporaryDirectory() as temp_path:

        # this is the new way of unpacking using the czann files
        model_metadata, model_path = DefaultConverter().unpack_model(model_file=czann_file, target_dir=Path(temp_path))

        # show model metadata
        print(model_metadata, "\n")

        # read individual model metadata
        print(model_metadata.model_type, "\n")

        # get model metadata as dictionary
        model_metadata_dict = model_metadata._asdict()

        for k, v in model_metadata_dict.items():
            print(k, "=", v)

        assert model_metadata.model_id == guid


@pytest.mark.parametrize(
    "czann, image, gpu, tiling, merge_window",
    [
        (
            "PGC_20X_nucleus_detector.czann",
            "PGC_20X.ome.tiff",
            False,
            TileMethod.CZTILE,
            SupportedWindow.none,
        ),
        (
            "PGC_20X_nucleus_detector.czann",
            "PGC_20X.ome.tiff",
            False,
            TileMethod.TILER,
            SupportedWindow.overlaptile,
        ),
    ],
)
@pytest.mark.skipif(not ONNXRUNTIME_AVAILABLE, reason="onnxruntime not available in CI environment")
def test_ndarray_prediction_seg(
    czann: str,
    image: str,
    gpu: bool,
    tiling: TileMethod,
    merge_window: SupportedWindow,
) -> None:
    """
    Test function for performing ndarray prediction segmentation.

    Args:
        czann (str): The path to the CZANN file.
        image (str): The path to the image file.
        gpu (bool): Flag indicating whether to use GPU for prediction.
        tiling (TileMethod): The tiling method to use.
        merge_window (SupportedWindow): Specifies which window function to use for Tiler only. Defaults to SupportedWindow.boxcar


    Returns:
        None
    """
    # get the correct file path for the sample data
    czann_file = get_testdata.get_modelfile(czann)
    image_file = get_testdata.get_imagefile(image)

    # read using BioImage
    bioio_img = BioImage(image_file)
    logger.info(f"Dimension Original Image: {bioio_img.dims}")
    logger.info(f"Array Shape Original Image: {bioio_img.shape}")

    # scale_x = 1.0
    # scale_y = 1.0

    # # Get physical pixel sizes using the correct API
    # scale_x = bioio_img.physical_pixel_sizes.X if bioio_img.physical_pixel_sizes.X is not None else 1.0
    # scale_y = bioio_img.physical_pixel_sizes.Y if bioio_img.physical_pixel_sizes.Y is not None else 1.0

    # Check dimensions and shape
    assert bioio_img.dims.order == "TCZYX"
    assert bioio_img.shape == (1, 1, 1, 2755, 3675)

    # Test the physical pixel sizes using the correct API
    assert bioio_img.physical_pixel_sizes.X == 0.227  # X
    assert bioio_img.physical_pixel_sizes.Y == 0.227  # Y
    assert bioio_img.physical_pixel_sizes.Z == 1.0  # Z

    # read the image data as numpy or dask array
    img = bioio_img.get_image_data()

    assert img.shape == (1, 1, 1, 2755, 3675)

    modeldata, seg_complete = predict.predict_ndarray(
        czann_file,
        img,
        border="auto",
        use_gpu=gpu,
        do_rescale=True,
        tiling_method=tiling,
        merge_window=merge_window,
    )

    assert seg_complete.shape == (1, 1, 1, 2755, 3675)
    assert seg_complete.ndim == 5  # Updated from 5 to 6 dimensions

    # create a list of label values
    label_values = list(range(1, len(modeldata.classes) + 1))

    # get individual outputs for all classes from the label image
    for c in range(len(modeldata.classes)):
        # get the pixels for which the value is equal to current class value
        print("Class Name:", modeldata.classes[c], "Prediction Pixel Value:", c)

        # get all pixels with a specific value as boolean array, convert to numpy array and label
        labels_current_class = process_nd.label_nd(seg_complete, labelvalue=label_values[c])
        print(f"Shape Labels Current Class: {labels_current_class.shape}")


@pytest.mark.parametrize(
    "czann, image, shape, gpu, tiling, merge_window",
    [
        (
            "simple_regmodel.czann",
            "LowSNR_s001.png",
            (1, 1, 1, 1024, 1024),
            False,
            TileMethod.CZTILE,
            SupportedWindow.none,
        ),
        (
            "N2V_tobacco_leaf.czann",
            "tobacco_leaf_WT_small.ome.tiff",
            (1, 1, 2, 1600, 1600),  # Updated to include extra dimension
            False,
            TileMethod.CZTILE,
            SupportedWindow.none,
        ),
        (
            "simple_regmodel.czann",
            "LowSNR_s001.png",
            (1, 1, 1, 1024, 1024),  # Updated to include extra dimension
            False,
            TileMethod.TILER,
            SupportedWindow.overlaptile,
        ),
        (
            "N2V_tobacco_leaf.czann",
            "tobacco_leaf_WT_small.ome.tiff",
            (1, 1, 2, 1600, 1600),  # Updated to include extra dimension
            False,
            TileMethod.TILER,
            SupportedWindow.overlaptile,
        ),
    ],
)
@pytest.mark.skipif(not ONNXRUNTIME_AVAILABLE, reason="onnxruntime not available in CI environment")
def test_ndarray_prediction_reg(
    czann: str,
    image: str,
    shape: Tuple[int, int, int, int, int],
    gpu: bool,
    tiling: TileMethod,
    merge_window: SupportedWindow,
) -> None:
    """
    Test the ndarray prediction using a regression model.

    Args:
        czann: The filename of the CZANN model.
        image: The filename of the image to be predicted.
        shape: The expected shape of the predicted segmentation.
        gpu: Whether to use GPU for prediction.
        tiling: The tiling method to be used.

    Returns:
        None
    """

    # get the correct file path for the sample data
    czann_file = get_testdata.get_modelfile(czann)
    image_file = get_testdata.get_imagefile(image)

    # read using BioImage
    bioio_img = BioImage(image_file)
    logger.info(f"Dimension Original Image: {bioio_img.dims}")
    logger.info(f"Array Shape Original Image: {bioio_img.shape}")

    # read the image data as numpy or dask array
    img = bioio_img.get_image_data()

    if img.shape[-1] == 3:
        img = img[..., 0]

    assert img.shape == shape

    modeldata, seg_complete = predict.predict_ndarray(
        czann_file,
        img,
        border="auto",
        use_gpu=gpu,
        do_rescale=False,
        tiling_method=tiling,
        merge_window=merge_window,
    )

    assert seg_complete.shape == shape
    assert seg_complete.ndim == 5

    print("Done.")


@pytest.mark.parametrize(
    "method",
    [
        (TileMethod.CZTILE),
    ],
)
def test_tiling_name(method):
    """
    Test the tiling name.

    Parameters:
    - method: The tiling method to be tested.

    Returns:
    - None

    Raises:
    - AssertionError: If the method is not equal to TileMethod.CZTILE.
    """
    assert method == TileMethod.CZTILE
