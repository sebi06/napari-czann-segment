# -*- coding: utf-8 -*-

#################################################################
# File        : predict.py
# Author      : sebi06, Team Enchilada
#
# Disclaimer: This code is purely experimental. Feel free to
# use it at your own risk.
#
#################################################################

import numpy as np
import tempfile
import itertools
from typing import List, Tuple, Dict, Union, Any, Optional
import dask.array as da
import torch
import onnxruntime as rt
from .onnx_inference import OnnxInferencer
from cztile.fixed_total_area_strategy import AlmostEqualBorderFixedTotalAreaStrategy2D
from cztile.tiling_strategy import Rectangle as czrect
from tqdm import tqdm
from czmodel.pytorch.convert import DefaultConverter
from pathlib import Path
import os


def predict_ndarray(czann_file: str,
                    img: Union[np.ndarray, da.Array],
                    border: Union[str, int] = "auto",
                    use_gpu: bool = False) -> Tuple[Any, Union[np.ndarray, da.Array]]:
    """Run the prediction on a multidimensional numpy array

    Args:
        czann_file (str): path for the *.czann file containing the ONNX model
        img (Union[np.ndarray, da.Array]): multi-dimensional array
        border (Union[str, int], optional): parameter to adjust the bordersize. Defaults to "auto".
        use_gpu (bool, optional): use GPU for the prediction. Defaults to False.

    Returns:
        Tuple[Any, Union[np.ndarray, da.Array]]: Return model metadata and the segmented multidimensional array
    """

    seg_complete = da.zeros_like(img, chunks=img.shape)

    # get the shape without the XY dimensions
    shape_woxy = img.shape[:-2]

    # create the "values" each for-loop iterates over
    loopover = [range(s) for s in shape_woxy]
    prod = itertools.product(*loopover)

    # extract the model information and path and to the prediction
    with tempfile.TemporaryDirectory() as temp_path:

        # this is the new way of unpacking using the czann files
        modelmd, model_path = DefaultConverter().unpack_model(model_file=czann_file,
                                                              target_dir=Path(temp_path))

        req_tilewidth = modelmd.input_shape[0]
        req_tileheight = modelmd.input_shape[1]

        print("Used TileSize: ", req_tilewidth, req_tileheight)

        # get the used bordersize - is needed for the tiling
        if isinstance(border, str) and border == "auto":
            # we assume same bordersize in XY
            bordersize = modelmd.min_overlap[0]
        else:
            bordersize = border

        print("Used Minimum BorderSize for Tiling: ", bordersize)

        # create ONNX inferencer once and use it for every tile
        inf = OnnxInferencer(str(model_path))

        # loop over all dimensions
        for idx in prod:

            # create list of slice-like objects based on the shape_woXY
            sl = len(shape_woxy) * [np.s_[0:1]]

            # insert the correct index into the respective slice objects for all dimensions
            for nd in range(len(shape_woxy)):
                sl[nd] = idx[nd]

            # extract the 2D image from the n-dimensional stack using the list of slice objects
            img2d = np.squeeze(img[tuple(sl)])

            # process the whole 2d image - make sure to use the correct **kwargs
            new_img2d = predict_tiles2d(img2d,
                                        model_path,
                                        inf=inf,
                                        tile_width=req_tilewidth,
                                        tile_height=req_tileheight,
                                        min_border_width=bordersize,
                                        use_gpu=use_gpu)

            # insert new 2D after tile-wise processing into nd array
            seg_complete[tuple(sl)] = new_img2d

    return modelmd, seg_complete


def predict_tiles2d(img2d: Union[np.ndarray, da.Array],
                    model_path: os.PathLike,
                    inf: OnnxInferencer,
                    tile_width: int = 1024,
                    tile_height: int = 1024,
                    min_border_width: int = 8,
                    do_rescale: bool = True,
                    use_gpu: bool = False) -> Union[np.ndarray, da.Array]:
    """Predict a larger 2D image array

    Args:
        img2d (Union[np.ndarray, da.Array]): larger 2D image
        model_path (os.PathLike): path to *.czann model file
        inf (OnnxInferencer): OnnxInferencer class to run the model
        tile_width (int, optional): width of tile required for prediction. Defaults to 1024.
        tile_height (int, optional): height of tile required for prediction. Defaults to 1024.
        min_border_width (int, optional): minimum border width for tiling. Defaults to 8.
        do_rescale (bool, optional): rescale the intensities [0-1]. Defaults to True.
        use_gpu (bool, optional): use GPU for the prediction. Defaults to False.

    Raises:
        tile_has_wrong_dimensionality: _description_

    Returns:
        Union[np.ndarray, da.Array]: segmented larger 2d image
    """

    if img2d.ndim == 2:

        new_img2d = da.zeros_like(img2d, chunks=(img2d.shape[0], img2d.shape[1]))

        # create a "tile" by specifying the desired tile dimension and the
        # minimum required overlap between tiles (depends on the processing)
        tiler = AlmostEqualBorderFixedTotalAreaStrategy2D(total_tile_width=tile_width,
                                                          total_tile_height=tile_height,
                                                          min_border_width=min_border_width)

        # create the tiles
        tiles = tiler.tile_rectangle(czrect(x=0, y=0, w=img2d.shape[0], h=img2d.shape[1]))

        # loop over all tiles
        for tile in tqdm(tiles):

            # get a single frame based on the roi
            tile2d = img2d[tile.roi.x:tile.roi.x + tile.roi.w, tile.roi.y:tile.roi.y + tile.roi.h]

            # make sure a numpy array is used for the prediction
            if isinstance(tile2d, da.Array):
                tile2d = tile2d.compute()

            if do_rescale:
                max_value = np.iinfo(tile2d.dtype).max
                tile2d = tile2d / (max_value - 1)

            # get the prediction for a single tile
            tile2d = inf.predict([tile2d[..., np.newaxis]], use_gpu=use_gpu)[0]

            # get the labels and add 1 to reflect the real values
            tile2d = np.argmax(tile2d, axis=-1) + 1

            # place result inside the new image
            new_img2d[tile.roi.x:tile.roi.x + tile.roi.w,
                      tile.roi.y:tile.roi.y + tile.roi.h] = tile2d

    else:
        raise tile_has_wrong_dimensionality(img2d.ndim)

    print("Datatype new_img2d", type(new_img2d))

    return new_img2d


def tile_has_wrong_dimensionality(num_dim: int) -> ValueError:
    """Check if the array as exactly 2 dimensions

    :param num_dim: number of dimensions
    :type num_dim: int
    :return: error message
    :rtype: ValueError
    """

    return ValueError(f"{str(num_dim)} does not equal 2.")
