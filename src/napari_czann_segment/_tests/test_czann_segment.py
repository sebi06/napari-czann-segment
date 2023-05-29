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
from napari_czann_segment import predict, process_nd
from aicsimageio import AICSImage
from pathlib import Path
import tempfile
import os
import torch
from czmodel.pytorch.convert import DefaultConverter
import pytest
from typing import List, Dict, Tuple, Optional, Type, Any, Union, Mapping


@pytest.mark.parametrize(
    "czann, guid",
    [
        ("PGC_20X_nucleus_detector.czann", "cd45c952-27d0-4f0f-888a-cf560ee5728f"),
        ("simple_regmodel_out5.czann", "b6f04bbd-2955-4996-8578-43d02d24f093")
    ]
)
def test_extract_model(czann: str, guid: str) -> None:

    # get data to test the functionality
    czann_file = get_testdata.get_modelfile(czann)

    # this is the old way to do it
    with tempfile.TemporaryDirectory() as temp_path:

        # this is the new way of unpacking using the czann files
        model_metadata, model_path = DefaultConverter().unpack_model(
            model_file=czann_file, target_dir=Path(temp_path))

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
    "czann, image, gpu",
    [
        ("PGC_20X_nucleus_detector.czann", "PGC_20X.ome.tiff", False),
    ]
)
def test_ndarray_prediction_seg(czann: str, image: str, gpu: bool) -> None:

    # get the correct file path for the sample data
    czann_file = get_testdata.get_modelfile(czann)
    image_file = get_testdata.get_imagefile(image)

    # read using AICSImageIO
    aics_img = AICSImage(image_file)
    print("Dimension Original Image:", aics_img.dims)
    print("Array Shape Original Image:", aics_img.shape)

    scale_x = 1.0
    scale_y = 1.0

    if aics_img.physical_pixel_sizes.X is not None:
        scale_x = aics_img.physical_pixel_sizes.X

    if aics_img.physical_pixel_sizes.X is None:
        scale_x = 1.0

    if aics_img.physical_pixel_sizes.Y is not None:
        scale_y = aics_img.physical_pixel_sizes.Y

    if aics_img.physical_pixel_sizes.X is None:
        scale_y = 1.0

    assert (aics_img.dims._dims_shape == {'T': 1,
                                          'C': 1,
                                          'Z': 1,
                                          'Y': 2755,
                                          'X': 3675})
    assert (aics_img.physical_pixel_sizes.X == 0.227)
    assert (aics_img.physical_pixel_sizes.Y == 0.227)
    assert (aics_img.physical_pixel_sizes.Z == 1.0)

    # get the scaling - will be applied to the segmentation outputs
    # scale = (aics_img.physical_pixel_sizes.X, aics_img.physical_pixel_sizes.Y)
    scale = (scale_x, scale_y)

    # read the image data as numpy or dask array
    img = aics_img.get_image_data()
    # img = aics_img.get_image_dask_data()

    assert (img.shape == (1, 1, 1, 2755, 3675))

    modeldata, seg_complete = predict.predict_ndarray(czann_file,
                                                      img,
                                                      border="auto",
                                                      use_gpu=gpu,
                                                      do_rescale=True)

    assert (seg_complete.shape == (1, 1, 1, 2755, 3675))
    assert (seg_complete.ndim == 5)
    assert (seg_complete.min().compute() == 1)
    assert (seg_complete.max().compute() == 2)

    # create a list of label values
    label_values = list(range(1, len(modeldata.classes) + 1))

    lc_min = [0, 0]
    lc_max = [56, 428]

    # get individual outputs for all classes from the label image
    for c in range(len(modeldata.classes)):
        # get the pixels for which the value is equal to current class value
        print("Class Name:", modeldata.classes[c], "Prediction Pixel Value:", c)

        # get all pixels with a specific value as boolean array, convert to numpy array and label
        labels_current_class = process_nd.label_nd(seg_complete,
                                                   labelvalue=label_values[c])

        assert (labels_current_class.min().compute() == lc_min[c])
        assert (labels_current_class.max().compute() == lc_max[c])

    print("Done.")


@pytest.mark.parametrize(
    "czann, image, shape, gpu",
    [
        ("simple_regmodel.czann", "LowSNR_s001.png", (1, 1, 1, 1024, 1024), False),
        ("N2V_tobacco_leaf.czann", "tobacco_leaf_WT_small.ome.tiff", (1, 1, 2, 1600, 1600), False)
    ]
)
def test_ndarray_prediction_reg(czann: str, image: str, shape: Tuple[int, int, int, int, int], gpu: bool) -> None:

    # get the correct file path for the sample data
    czann_file = get_testdata.get_modelfile(czann)
    image_file = get_testdata.get_imagefile(image)

    # read using AICSImageIO
    aics_img = AICSImage(image_file)
    print("Dimension Original Image:", aics_img.dims)
    print("Array Shape Original Image:", aics_img.shape)

    # read the image data as numpy or dask array
    img = aics_img.get_image_data()

    if img.shape[-1] == 3:
        img = img[..., 0]

    assert (img.shape == shape)

    modeldata, seg_complete = predict.predict_ndarray(czann_file,
                                                      img,
                                                      border="auto",
                                                      use_gpu=gpu,
                                                      do_rescale=False)

    assert (seg_complete.shape == shape)
    assert (seg_complete.ndim == 5)

    print("Done.")
