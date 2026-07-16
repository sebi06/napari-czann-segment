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
import math
from typing import Tuple, Union, Any, List, Optional
import dask.array as da
from .onnx_inference import OnnxInferencer
from czmodel import ModelType, ModelMetadata
from cztile.fixed_total_area_strategy_2d import (
    AlmostEqualBorderFixedTotalAreaStrategy2D,
)

from cztile.tiling_strategy import Region2D, TileInput
from tqdm import tqdm, trange
from tiler import Tiler, Merger

from czmodel.core.util._extract_model import extract_czann_model
from napari_czann_segment.czseg_parser import extract_czseg_model
from pathlib import Path
from .utils import TileMethod, SupportedWindow
from ryomen import Slicer
from .utils import setup_log
import xarray as xr

logger = setup_log("Napari-CZANN-predict")


def _resolve_do_rescale(value: Optional[bool]) -> bool:
    """Resolve missing model scaling metadata to the legacy default."""
    return True if value is None else bool(value)


def predict_ndarray(
    czann_file: str,
    img: Union[np.ndarray, da.Array, xr.DataArray],
    border: Union[str, int] = "auto",
    use_gpu: bool = False,
    do_rescale: Optional[bool] = None,
    tiling_method: TileMethod = TileMethod.CZTILE,
    merge_window: SupportedWindow = SupportedWindow.none,
    batch_size: int = 8,
    convert_rgb_to_bgr: Optional[bool] = None,
) -> Tuple[Any, Union[np.ndarray, da.Array]]:
    """Run the prediction on a multidimensional numpy array

    Args:
        czann_file (str): path for the *.czann file containing the ONNX model
        img (Union[np.ndarray, da.Array]): multi-dimensional array
        border (Union[str, int], optional): parameter to adjust the bordersize. Defaults to "auto".
        use_gpu (bool, optional): use GPU for the prediction. Defaults to False
        do_rescale (bool, optional): rescale the intensities [0-1]. If None,
            use the model metadata. Defaults to None.
        tiling_method (TileMethod, optional): specify the desired tiling method. Defaults to TileMethod.CZITILE
        merge_window (SupportedWindow, optional): Specifies which window function to use for Tiler only. Defaults to SupportedWindow.boxcar
        batch_size (int, optional): batch size for inference (higher values improve GPU utilization). Defaults to 8.
        convert_rgb_to_bgr (Optional[bool], optional): convert RGB image to BGR format (for models expecting BGR).
            If None, auto-detect from the model metadata (ZEN color models trained on Bgr24 expect BGR while
            czitools delivers RGB). Defaults to None.

    Returns:
        Tuple[Any, Union[np.ndarray, da.Array]]: Return model metadata and the segmented multidimensional array
    """

    # convert xarray to numpy array
    if isinstance(img, xr.DataArray):
        img = img.data

    # extract the model information and path and to the prediction
    with tempfile.TemporaryDirectory() as temp_path:

        # Determine file type and use appropriate parser
        file_extension = Path(czann_file).suffix.lower()

        # Whether the model was trained on BGR-ordered channels. czitools converts
        # BGR CZI data to RGB on read, so RGB->BGR conversion is required for these
        # models to match the training/inference pipeline of ZEN / SegmentationService.
        model_expects_bgr = False

        if file_extension == ".czseg":
            logger.info("Detected CZSEG file format")
            modelmd, model_path, model_expects_bgr, _ = extract_czseg_model(path=czann_file, target_dir=Path(temp_path))
        elif file_extension in [".czann", ".czmodel"]:
            logger.info(f"Detected {file_extension.upper()} file format")
            modelmd, model_path = extract_czann_model(path=czann_file, target_dir=Path(temp_path))
        else:
            raise ValueError(f"Unsupported model file format: {file_extension}")

        # Resolve RGB->BGR conversion from model metadata when not set explicitly.
        if convert_rgb_to_bgr is None:
            convert_rgb_to_bgr = model_expects_bgr

        if do_rescale is None:
            do_rescale = _resolve_do_rescale(getattr(modelmd, "scaling", None))

        input_channels = modelmd.input_shape[-1]

        # Determine if input has an explicit channel dimension at the end.
        # Expected format: [..., Y, X, C], where C matches the model input.
        # This includes plain color images shaped (Y, X, 3).
        has_channel_dim = (
            len(img.shape) >= 3 and img.shape[-1] == input_channels and (input_channels > 1 or len(img.shape) == 3)
        )

        if has_channel_dim:
            shape_woxy = img.shape[:-3]  # Remove Y, X, C dimensions
            # Output shape is [..., Y, X] (no channel dimension in output - it's a label map)
            output_shape = img.shape[:-1]  # Remove only the channel dimension
        else:
            shape_woxy = img.shape[:-2]  # Remove Y, X dimensions
            # Output shape same as input for 2D images
            output_shape = img.shape

        # seg_complete will hold the 2D segmentation/regression output
        seg_complete = np.zeros(output_shape, dtype=img.dtype)

        # DEBUG: Log image information
        logger.info(f"[predict_ndarray] Input image shape: {img.shape}, dtype: {img.dtype}")
        logger.info(f"[predict_ndarray] Has channel dim: {has_channel_dim}, Output shape: {output_shape}")
        logger.info(f"[predict_ndarray] do_rescale value: {do_rescale}")
        logger.info(f"[predict_ndarray] convert_rgb_to_bgr parameter value: {convert_rgb_to_bgr}")

        # create the "values" each for-loop iterates over
        loopover = [range(s) for s in shape_woxy]
        prod = itertools.product(*loopover)

        # get the used bordersize - is needed for the tiling
        if isinstance(border, str) and border == "auto":
            # we assume same bordersize in XY
            bordersize = modelmd.min_overlap[0]
        else:
            bordersize = border

        # create ONNX inferencer once and use it for every tile
        inf = OnnxInferencer(str(model_path), batch_size=batch_size)

        if convert_rgb_to_bgr:
            logger.info("RGB to BGR conversion: ENABLED")
        else:
            logger.info("RGB to BGR conversion: DISABLED")

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
                convert_rgb_to_bgr=convert_rgb_to_bgr,
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
    convert_rgb_to_bgr: bool = False,
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
        convert_rgb_to_bgr (bool, optional): convert RGB image to BGR format (for models expecting BGR). Defaults to False.

    Raises:
        tile_has_wrong_dimensionality: raised if a tile has the wrong dimensionality

    Returns:
        Union[np.ndarray, da.Array]: segmented larger 2d image
    """

    # Handle both 2D grayscale (Y, X) and 3D color (Y, X, C) images
    if img2d.ndim == 2:
        # Grayscale image: shape is (Y, X)
        height, width = img2d.shape
        channels = None
    elif img2d.ndim == 3:
        # Color/multi-channel image: shape is (Y, X, C)
        height, width, channels = img2d.shape
    else:
        raise tile_has_wrong_dimensionality(img2d.ndim)

    # Output is always 2D (Y, X) even if input has channels, because it's a label/regression map
    new_img2d = np.zeros((height, width), dtype=img2d.dtype)

    input_channels = model_md.input_shape[-1]
    batch_sz = inferencer.batch_size

    # DEBUG: Log tiling parameters
    logger.info(f"[predict_tiles2d] Input image 2D shape: {img2d.shape}, dtype: {img2d.dtype}")
    logger.info(f"[predict_tiles2d] Model expects {input_channels} channels, model input shape: {model_md.input_shape}")
    logger.info(f"[predict_tiles2d] Tiling method: {tiling_method}")
    logger.info(f"[predict_tiles2d] convert_rgb_to_bgr parameter: {convert_rgb_to_bgr}")
    if convert_rgb_to_bgr:
        logger.warning(f"[predict_tiles2d] >>> RGB to BGR CONVERSION IS ENABLED <<<")

    # ------------------------------------------------------------------
    # Helper: preprocess one extracted tile array into a float32 tensor
    # with shape (H, W, C) ready for inference.
    # ------------------------------------------------------------------
    def _prep(t: np.ndarray) -> np.ndarray:
        if isinstance(t, da.Array):
            t = t.compute()
        if do_rescale:
            t = t.astype(np.float32) / np.iinfo(img2d.dtype).max
        else:
            t = t.astype(np.float32)
        if t.ndim == 2:
            t = t[..., np.newaxis]
        if t.shape[-1] != input_channels:
            raise ValueError(
                f"Channel mismatch: tile has {t.shape[-1]} channel(s), " f"but model expects {input_channels}."
            )
        # Convert RGB to BGR if requested (for models trained on BGR format)
        if convert_rgb_to_bgr and t.shape[-1] == 3:
            # Try the opposite direction first to debug
            logger.debug(f"Before conversion: channels = {t.shape[-1]}, dtype = {t.dtype}")
            t_orig = t.copy()
            t = t[..., ::-1]  # Reverse channels
            logger.debug(
                f"After conversion: original[:, :, 0] mean={t_orig[..., 0].mean()}, converted[:, :, 0] mean={t[..., 0].mean()}"
            )
        return t

    # ------------------------------------------------------------------
    # Helper: extract raw tile from img2d given a cztile roi
    # ------------------------------------------------------------------
    def _extract(roi) -> np.ndarray:
        if channels is None:
            return img2d[roi.y : roi.y + roi.h, roi.x : roi.x + roi.w]
        return img2d[roi.y : roi.y + roi.h, roi.x : roi.x + roi.w, :]

    # DEBUG: Log what channels value is
    logger.info(f"[predict_tiles2d] Image has channels={channels}, so tiles will have shape (H, W, {channels})")
    if channels == 3:
        logger.warning(f"[predict_tiles2d] Confirmed: 3-channel image detected, conversion WILL be applied if enabled")

    if tiling_method is TileMethod.CZTILE:

        # ------------------------------------------------------------------
        # CZTILE tiling (default, mirrors how ZEN / SegmentationService tile).
        #
        # Each cztile ``Tile2D`` describes three related regions:
        #   * ``tile.roi``    - the FULL tile INCLUDING its border. This is the
        #                       region that is cut out of the image and fed to the
        #                       model. Its size equals the model input size
        #                       (e.g. 256 x 256).
        #   * ``tile.center`` - the INTERIOR region this tile is responsible for.
        #                       The center regions of all tiles tessellate the
        #                       image EXACTLY (no overlap).
        #   * ``tile.border`` - the per-side border widths (left/top/right/bottom)
        #                       so that ``roi = center + border`` on every side.
        #
        # The border only exists to give the center enough receptive-field context
        # for the model; it MUST be discarded after inference. We therefore feed
        # ``tile.roi`` to the model but write back only ``tile.center`` (see below).
        # This matches ZEN's C# pipeline (RestSegmenter + output assembler), which
        # also predicts on the bordered tile and stores only the interior.
        # ------------------------------------------------------------------
        tiler = AlmostEqualBorderFixedTotalAreaStrategy2D(
            width=TileInput(model_md.input_shape[0], min_border_length=min_border_width),
            height=TileInput(model_md.input_shape[1], min_border_length=min_border_width),
        )
        region2d = Region2D(x=0, y=0, w=width, h=height)
        tiles = list(tiler.calculate_2d_tiles(region2d=region2d))
        n_batches = math.ceil(len(tiles) / batch_sz)

        logger.info(f"[CZTILE] Total tiles: {len(tiles)}, Batches: {n_batches}, Batch size: {batch_sz}")
        logger.info(
            f"[CZTILE] Image dimensions: {height} x {width}, Model expects: {model_md.input_shape[0]} x {model_md.input_shape[1]}"
        )
        logger.info(f"[CZTILE] min_border_width (overlap): {min_border_width}")

        # Log first few tiles to verify they make sense
        if len(tiles) > 0:
            logger.info(
                f"[CZTILE] First tile ROI: x={tiles[0].roi.x}, y={tiles[0].roi.y}, w={tiles[0].roi.w}, h={tiles[0].roi.h}"
            )
        if len(tiles) > 1:
            logger.info(
                f"[CZTILE] Second tile ROI: x={tiles[1].roi.x}, y={tiles[1].roi.y}, w={tiles[1].roi.w}, h={tiles[1].roi.h}"
            )
        if len(tiles) > 2:
            logger.info(
                f"[CZTILE] Third tile ROI: x={tiles[2].roi.x}, y={tiles[2].roi.y}, w={tiles[2].roi.w}, h={tiles[2].roi.h}"
            )

        with tqdm(total=len(tiles), desc="Tiles (CZTILE)", unit="tile") as pbar:
            for batch_start in range(0, len(tiles), batch_sz):
                batch_tiles = tiles[batch_start : batch_start + batch_sz]
                # Cut the FULL bordered tile (t.roi) out of the image for inference.
                batch_imgs = [_prep(_extract(t.roi)) for t in batch_tiles]

                # DEBUG: Log first batch tile shape
                if batch_start == 0:
                    first_tile = _extract(batch_tiles[0].roi)
                    logger.debug(f"[CZTILE] First tile extracted shape: {first_tile.shape}, dtype: {first_tile.dtype}")
                    first_tile_prepped = _prep(first_tile)
                    logger.debug(
                        f"[CZTILE] First tile after _prep: {first_tile_prepped.shape}, dtype: {first_tile_prepped.dtype}"
                    )

                if model_md.model_type == ModelType.SINGLE_CLASS_SEMANTIC_SEGMENTATION:
                    results = inferencer.predict(batch_imgs, use_gpu=use_gpu)
                    for t, res in zip(batch_tiles, results):
                        # ``res`` is the prediction for the FULL bordered tile
                        # (same H x W as t.roi). Convert the per-class probabilities
                        # to a label map (argmax) and add 1 so labels are 1-based.
                        labels = np.argmax(res, axis=-1) + 1

                        # Crop the border away to keep only the tile's CENTER.
                        # Inside the tile, the center starts at (border.top,
                        # border.left) and has size (center.h, center.w). Because
                        # roi = center + border, this offset selects exactly the
                        # interior pixels of the prediction.
                        cy0, cx0 = t.border.top, t.border.left
                        center_labels = labels[cy0 : cy0 + t.center.h, cx0 : cx0 + t.center.w]

                        # Write the center to its (non-overlapping) location in the
                        # output. Since tile centers tessellate the image, no two
                        # tiles write to the same pixel -> no border seams.
                        new_img2d[
                            t.center.y : t.center.y + t.center.h,
                            t.center.x : t.center.x + t.center.w,
                        ] = center_labels

                elif model_md.model_type == ModelType.REGRESSION:
                    results = inferencer.predict(batch_imgs, use_gpu=use_gpu)
                    for t, res in zip(batch_tiles, results):
                        # Regression output has a single channel; apply the exact
                        # same border-crop / center-only write-back as above so the
                        # regressed map is seam-free.
                        cy0, cx0 = t.border.top, t.border.left
                        center_values = res[cy0 : cy0 + t.center.h, cx0 : cx0 + t.center.w, 0]
                        new_img2d[
                            t.center.y : t.center.y + t.center.h,
                            t.center.x : t.center.x + t.center.w,
                        ] = center_values

                pbar.update(len(batch_tiles))

    if tiling_method is TileMethod.TILER:

        # ------------------------------------------------------------------
        # TILER tiling (via the third-party ``tiler`` package).
        #
        # Seam handling differs from CZTILE: instead of cropping each tile to a
        # non-overlapping center, ``tiler`` keeps overlapping tiles and blends
        # them in the ``Merger`` using a window function (``merge_window``):
        #   * "boxcar"      - uniform weights (plain average of the overlap).
        #   * "overlap-tile"- higher weight towards each tile center, so border
        #                     predictions contribute less -> smooth transitions.
        # The windowed blend is what prevents hard border seams here; there is no
        # explicit center crop because ``Merger.merge(unpad=True)`` already removes
        # the padding added for edge tiles.
        #
        # NOTE: this path currently supports single-channel (grayscale) input only.
        # For multi-channel (e.g. BGR) images ``tiler`` raises a shape error, so
        # color models must use the CZTILE method.
        # ------------------------------------------------------------------
        if merge_window is SupportedWindow.overlaptile:
            merge_window_name = "overlap-tile"
        elif merge_window is SupportedWindow.none:
            merge_window_name = "boxcar"
        else:
            merge_window_name = merge_window.name

        channel_dim = 2 if channels is not None else None

        tiler = Tiler(
            data_shape=img2d.shape,
            tile_shape=(model_md.input_shape[0], model_md.input_shape[1]),
            overlap=(min_border_width, min_border_width),
            channel_dimension=channel_dim,
            mode="reflect",
        )

        if model_md.model_type == ModelType.SINGLE_CLASS_SEMANTIC_SEGMENTATION:
            merger = Merger(tiler, window=merge_window_name)

            with tqdm(total=tiler.n_tiles, desc="Tiles (TILER)", unit="tile") as pbar:
                for batch_start in range(0, tiler.n_tiles, batch_sz):
                    ids = list(range(batch_start, min(batch_start + batch_sz, tiler.n_tiles)))
                    batch_imgs = [_prep(tiler.get_tile(img2d, i)) for i in ids]
                    results = inferencer.predict(batch_imgs, use_gpu=use_gpu)
                    for tile_id, res in zip(ids, results):
                        merger.add(tile_id, np.argmax(res, axis=-1) + 1)
                    pbar.update(len(ids))

            new_img2d = merger.merge(unpad=True)

        elif model_md.model_type == ModelType.REGRESSION:
            merger = Merger(tiler, window=merge_window_name)

            with tqdm(total=tiler.n_tiles, desc="Tiles (TILER)", unit="tile") as pbar:
                for batch_start in range(0, tiler.n_tiles, batch_sz):
                    ids = list(range(batch_start, min(batch_start + batch_sz, tiler.n_tiles)))
                    batch_imgs = [_prep(tiler.get_tile(img2d, i)) for i in ids]
                    results = inferencer.predict(batch_imgs, use_gpu=use_gpu)
                    for tile_id, res in zip(ids, results):
                        merger.add(tile_id, res[..., 0])
                    pbar.update(len(ids))

            new_img2d = merger.merge(unpad=True)

    if tiling_method is TileMethod.RYOMEN:

        # ------------------------------------------------------------------
        # RYOMEN tiling (via the third-party ``ryomen`` package).
        #
        # ``ryomen.Slicer`` yields, for every tile, a triple ``(tile, src, dst)``:
        #   * ``tile`` - the padded/overlapping crop fed to the model.
        #   * ``src``  - the slice INSIDE the tile that maps to the unique output
        #                region (i.e. the center, with the overlap removed).
        #   * ``dst``  - the matching (non-overlapping) slice in the output image.
        # Writing ``prediction[src] -> new_img2d[dst]`` therefore stores only each
        # tile's center, exactly like the CZTILE border-crop above, so this method
        # is seam-free by construction (no blending needed).
        #
        # NOTE: like TILER, this path currently supports single-channel input only;
        # for multi-channel images the Slicer treats the channel axis as spatial
        # and fails. Use CZTILE for color models.
        # ------------------------------------------------------------------
        slices = Slicer(
            img2d,
            crop_size=(model_md.input_shape[0], model_md.input_shape[1]),
            overlap=(min_border_width, min_border_width),
            pad=True,
        )
        all_slices = list(slices)  # [(tile, source, destination), ...]

        if model_md.model_type == ModelType.SINGLE_CLASS_SEMANTIC_SEGMENTATION:
            with tqdm(total=len(all_slices), desc="Tiles (RYOMEN)", unit="tile") as pbar:
                for batch_start in range(0, len(all_slices), batch_sz):
                    batch = all_slices[batch_start : batch_start + batch_sz]
                    batch_imgs = [_prep(t) for t, _, _ in batch]
                    results = inferencer.predict(batch_imgs, use_gpu=use_gpu)
                    for (_, src, dst), res in zip(batch, results):
                        new_img2d[dst] = (np.argmax(res, axis=-1) + 1)[src]
                    pbar.update(len(batch))

        elif model_md.model_type == ModelType.REGRESSION:
            with tqdm(total=len(all_slices), desc="Tiles (RYOMEN)", unit="tile") as pbar:
                for batch_start in range(0, len(all_slices), batch_sz):
                    batch = all_slices[batch_start : batch_start + batch_sz]
                    batch_imgs = [_prep(t) for t, _, _ in batch]
                    results = inferencer.predict(batch_imgs, use_gpu=use_gpu)
                    for (_, src, dst), res in zip(batch, results):
                        new_img2d[dst] = res[..., 0][src]
                    pbar.update(len(batch))

    return new_img2d


def process_semantic(
    tile2d: Union[np.ndarray, da.Array],
    inferencer: OnnxInferencer,
    use_gpu: bool = False,
    do_rescale: bool = True,
    input_channels: int = 1,
):
    """
    Process the semantic segmentation for a given 2D tile.

    Args:
        tile2d (Union[np.ndarray, da.Array]): The input 2D tile for semantic segmentation.
            Can be (H, W) or (H, W, C) where C is the channel dimension.
        inferencer (OnnxInferencer): The inferencer object used for prediction.
        use_gpu (bool, optional): Whether to use GPU for prediction. Defaults to False.
        do_rescale (bool, optional): Whether to rescale the input tile. Defaults to True.
        input_channels (int, optional): Expected number of input channels from model. Defaults to 1.

    Returns:
        np.ndarray: The processed semantic segmentation result for the input tile.
    """

    # make sure a numpy array is used for the prediction
    if isinstance(tile2d, da.Array):
        tile2d = tile2d.compute()

    if do_rescale:
        max_value = np.iinfo(tile2d.dtype).max
        tile2d = tile2d / (max_value - 1)

    # Handle channel dimension
    # If tile has shape (H, W), add channel dimension
    if tile2d.ndim == 2:
        tile2d = tile2d[..., np.newaxis]

    # Now tile2d has shape (H, W, C) where C is current channels
    current_channels = tile2d.shape[-1]

    # Verify channel count matches what the model expects
    if current_channels != input_channels:
        raise ValueError(
            f"Channel mismatch: input image has {current_channels} channel(s), "
            f"but model expects {input_channels}. "
            f"Please provide an image whose channel count matches the model."
        )

    # get the prediction for a single tile
    tile2d = inferencer.predict([tile2d], use_gpu=use_gpu)[0]

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
