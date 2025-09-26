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
from typing import Tuple, Union, Any
import dask.array as da
from .onnx_inference import OnnxInferencer
from czmodel import ModelType, ModelMetadata
from cztile.fixed_total_area_strategy import (
    AlmostEqualBorderFixedTotalAreaStrategy2D,
)
from cztile.tiling_strategy import Rectangle as czrect
from tqdm import tqdm, trange
from tiler import Tiler, Merger
from czmodel.pytorch.convert import DefaultConverter
from pathlib import Path
from .utils import TileMethod, SupportedWindow
from ryomen import Slicer
from .utils import setup_log


logger = setup_log("Napari-CZANN-predict")


def predict_ndarray(
    czann_file: str,
    img: Union[np.ndarray, da.Array],
    border: Union[str, int] = "auto",
    use_gpu: bool = False,
    do_rescale: bool = True,
    tiling_method: TileMethod = TileMethod.CZTILE,
    merge_window: SupportedWindow = SupportedWindow.none,
) -> Tuple[Any, Union[np.ndarray, da.Array]]:
    """Run the prediction on a multidimensional numpy array

    Args:
        czann_file (str): path for the *.czann file containing the ONNX model
        img (Union[np.ndarray, da.Array]): multi-dimensional array
        border (Union[str, int], optional): parameter to adjust the bordersize. Defaults to "auto".
        use_gpu (bool, optional): use GPU for the prediction. Defaults to False
        do_rescale (bool, optional): rescale the intensities [0-1]. Defaults to True.
        tiling_method (TileMethod, optional): specify the desired tiling method. Defaults to TileMethod.CZITILE
        merge_window (SupportedWindow, optional): Specifies which window function to use for Tiler only. Defaults to SupportedWindow.boxcar

    Returns:
        Tuple[Any, Union[np.ndarray, da.Array]]: Return model metadata and the segmented multidimensional array
    """

    # seg_complete = da.zeros_like(img, chunks=img.shape)
    seg_complete = np.zeros_like(img)

    # get the shape without the XY dimensions
    shape_woxy = img.shape[:-2]

    # create the "values" each for-loop iterates over
    loopover = [range(s) for s in shape_woxy]
    prod = itertools.product(*loopover)

    # extract the model information and path and to the prediction
    with tempfile.TemporaryDirectory() as temp_path:

        # this is the new way of unpacking using the czann files
        modelmd, model_path = DefaultConverter().unpack_model(
            model_file=czann_file, target_dir=Path(temp_path)
        )

        # get the used bordersize - is needed for the tiling
        if isinstance(border, str) and border == "auto":
            # we assume same bordersize in XY
            bordersize = modelmd.min_overlap[0]
        else:
            bordersize = border

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
            new_img2d = predict_tiles2d(
                img2d,
                model_md=modelmd,
                inferencer=inf,
                min_border_width=bordersize,
                do_rescale=do_rescale,
                use_gpu=use_gpu,
                tiling_method=tiling_method,
                merge_window=merge_window,
            )

            # insert new 2D after tile-wise processing into nd array
            seg_complete[tuple(sl)] = new_img2d

    return modelmd, seg_complete


def predict_tiles2d(
    img2d: Union[np.ndarray, da.Array],
    model_md: ModelMetadata,
    inferencer: OnnxInferencer,
    min_border_width: int = 8,
    do_rescale: bool = True,
    use_gpu: bool = False,
    tiling_method: TileMethod = TileMethod.CZTILE,
    merge_window: SupportedWindow = SupportedWindow.none,
) -> Union[np.ndarray, da.Array]:
    """Predict a larger 2D image array

    Args:
        img2d (Union[np.ndarray, da.Array]): larger 2D image
        model_md (ModelMetadata): The metadata for this model
        inferencer (OnnxInferencer): OnnxInferencer class to run the model
        tile_width (int, optional): width of tile required for prediction. Defaults to 1024.
        tile_height (int, optional): height of tile required for prediction. Defaults to 1024.
        min_border_width (int, optional): minimum border width for tiling. Defaults to 8.
        do_rescale (bool, optional): rescale the intensities [0-1]. Defaults to True.
        use_gpu (bool, optional): use GPU for the prediction. Defaults to False.
        tiling_method (TileMethod, optional): specify the desired tiling method. Defaults to TileMethod.CZITILE.
        merge_window (SupportedWindow, optional): Specifies which window function to use for Tiler only. Defaults to SupportedWindow.boxcar.

    Raises:
        tile_has_wrong_dimensionality: raised if a tile has the wrong dimensionality

    Returns:
        Union[np.ndarray, da.Array]: segmented larger 2d image
    """

    if img2d.ndim == 2:

        # new_img2d = da.zeros_like(img2d, chunks=(img2d.shape[0], img2d.shape[1]))
        new_img2d = np.zeros_like(img2d)

        if tiling_method is TileMethod.CZTILE:

            # create a "tile" by specifying the desired tile dimension and the
            # minimum required overlap between tiles (depends on the processing)

            tiler = AlmostEqualBorderFixedTotalAreaStrategy2D(
                total_tile_width=model_md.input_shape[0],
                total_tile_height=model_md.input_shape[1],
                min_border_width=min_border_width,
            )

            # create the tiles
            tiles = tiler.tile_rectangle(
                czrect(x=0, y=0, w=img2d.shape[0], h=img2d.shape[1])
            )

            # loop over all tiles
            for tile in tqdm(tiles):

                # get a single frame based on the roi
                tile2d = img2d[
                    tile.roi.x : tile.roi.x + tile.roi.w,
                    tile.roi.y : tile.roi.y + tile.roi.h,
                ]

                # run the prediction
                if (
                    model_md.model_type
                    == ModelType.SINGLE_CLASS_SEMANTIC_SEGMENTATION
                ):

                    tile2d = process_semantic(
                        tile2d,
                        inferencer=inferencer,
                        use_gpu=use_gpu,
                        do_rescale=do_rescale,
                    )

                    # place result inside the new image
                    new_img2d[
                        tile.roi.x : tile.roi.x + tile.roi.w,
                        tile.roi.y : tile.roi.y + tile.roi.h,
                    ] = tile2d

                if model_md.model_type == ModelType.REGRESSION:

                    # get the prediction for a single tile
                    tile2d = inferencer.predict(
                        [tile2d[..., np.newaxis]], use_gpu=use_gpu
                    )[0]

                    # place result inside the new image
                    new_img2d[
                        tile.roi.x : tile.roi.x + tile.roi.w,
                        tile.roi.y : tile.roi.y + tile.roi.h,
                    ] = tile2d[..., 0]

        if tiling_method is TileMethod.TILER:

            if merge_window is SupportedWindow.overlaptile:
                merge_window_name = "overlap-tile"
            else:
                merge_window_name = merge_window.name

            if merge_window is SupportedWindow.none:
                merge_window_name = "boxcar"

            tiler = Tiler(
                data_shape=img2d.shape,
                tile_shape=(model_md.input_shape[0], model_md.input_shape[1]),
                overlap=(min_border_width, min_border_width),
                channel_dimension=None,
                mode="reflect",
            )

            # Setup merging parameters
            if (
                model_md.model_type
                == ModelType.SINGLE_CLASS_SEMANTIC_SEGMENTATION
            ):
                logger.info(f"Using Merging Window: {merge_window_name}")
                merger = Merger(tiler, window=merge_window_name)

                for tile_id in trange(tiler.n_tiles):
                    tile2d = tiler.get_tile(img2d, tile_id)

                    # do the processing
                    tile2d = process_semantic(
                        tile2d,
                        inferencer=inferencer,
                        use_gpu=use_gpu,
                        do_rescale=do_rescale,
                    )

                    merger.add(tile_id, tile2d)

                new_img2d = merger.merge(unpad=True)

            if model_md.model_type == ModelType.REGRESSION:
                merger = Merger(tiler, window=merge_window_name)

                for tile_id in trange(tiler.n_tiles):
                    tile2d = tiler.get_tile(img2d, tile_id)

                    # get the prediction for a single tile
                    tile2d = inferencer.predict(
                        [tile2d[..., np.newaxis]], use_gpu=use_gpu
                    )[0]

                    merger.add(tile_id, tile2d[..., 0])

                new_img2d = merger.merge(unpad=True)

        if tiling_method is TileMethod.RYOMEN:

            slices = Slicer(
                img2d,
                crop_size=(model_md.input_shape[0], model_md.input_shape[1]),
                overlap=(min_border_width, min_border_width),
                pad=True,
            )

            # Setup merging parameters
            if (
                model_md.model_type
                == ModelType.SINGLE_CLASS_SEMANTIC_SEGMENTATION
            ):

                for tile2d, source, destination in tqdm(slices):

                    tile2d = process_semantic(
                        tile2d,
                        inferencer=inferencer,
                        use_gpu=use_gpu,
                        do_rescale=do_rescale,
                    )

                    new_img2d[destination] = tile2d[source]

            if model_md.model_type == ModelType.REGRESSION:

                for tile2d, source, destination in tqdm(slices):

                    # get the prediction for a single tile
                    tile2d = inferencer.predict(
                        [tile2d[..., np.newaxis]], use_gpu=use_gpu
                    )[0]

                    new_img2d[destination] = tile2d[source]

    else:
        raise tile_has_wrong_dimensionality(img2d.ndim)

    return new_img2d


def process_semantic(
    tile2d: Union[np.ndarray, da.Array],
    inferencer: OnnxInferencer,
    use_gpu: bool = False,
    do_rescale: bool = True,
):
    """
    Process the semantic segmentation for a given 2D tile.

    Args:
        tile2d (Union[np.ndarray, da.Array]): The input 2D tile for semantic segmentation.
        inferencer (OnnxInferencer): The inferencer object used for prediction.
        use_gpu (bool, optional): Whether to use GPU for prediction. Defaults to False.
        do_rescale (bool, optional): Whether to rescale the input tile. Defaults to True.

    Returns:
        np.ndarray: The processed semantic segmentation result for the input tile.
    """

    # make sure a numpy array is used for the prediction
    if isinstance(tile2d, da.Array):
        tile2d = tile2d.compute()

    if do_rescale:
        max_value = np.iinfo(tile2d.dtype).max
        tile2d = tile2d / (max_value - 1)

    # get the prediction for a single tile
    tile2d = inferencer.predict([tile2d[..., np.newaxis]], use_gpu=use_gpu)[0]

    # get the labels and add 1 to reflect the real values
    tile2d = np.argmax(tile2d, axis=-1) + 1

    return tile2d


def tile_has_wrong_dimensionality(num_dim: int) -> ValueError:
    """Check if the array has exactly 2 dimensions.

    :param num_dim: The number of dimensions in the array.
    :type num_dim: int
    :return: A ValueError with an error message.
    :rtype: ValueError
    """
    return ValueError(f"{str(num_dim)} does not equal 2.")
