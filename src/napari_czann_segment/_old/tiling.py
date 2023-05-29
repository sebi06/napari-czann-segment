# -*- coding: utf-8 -*-

#################################################################
# File        : tiling.py
# Author      : sebi06, Team Enchilada
#
# Disclaimer: This code is purely experimental. Feel free to
# use it at your own risk.
#
#################################################################

import numpy as np
from cztile.fixed_total_area_strategy import AlmostEqualBorderFixedTotalAreaStrategy2D
from cztile.tiling_strategy import Tile2D
from cztile.tiling_strategy import Rectangle as czrect
from typing import List, NamedTuple, Union, Tuple, Callable
import dask.array as da
from tqdm import tqdm


def process2d_tiles(func2d: Callable,
                    img2d: Union[np.ndarray, da.Array],
                    tile_width: int = 64,
                    tile_height: int = 64,
                    min_border_width: int = 8,
                    use_dask: bool = True,
                    **kwargs: int) -> Union[np.ndarray, da.Array]:
    """Function to process a "larger" 2d image using a 2d processing function
    that will be applied to the "larger" image in a tile-wise manner

    :param func2d: 2d processing function that will be applied tile-wise
    :type func2d: Callable
    :param img2d: the "larger" 2d image to be processed
    :type img2d: Union[np.ndarray, da.array]
    :param tile_width: tile width in pixel, defaults to 64
    :type tile_width: int, optional
    :param tile_height: tile height in pixel, defaults to 64
    :type tile_height: int, optional
    :param min_border_width: minimum border width in pixel, defaults to 8
    :type min_border_width: int, optional
    :param use_dask: out will be dask array, defaults to True
    :type use_dask: bool
    :return: the processed "larger" 2d image
    :rtype: Union[np.ndarray, da.array]
    """

    if img2d.ndim == 2:

        if use_dask:
            new_img2d = da.zeros_like(img2d, chunks=(img2d.shape[0], img2d.shape[1]))
        if not use_dask:
            new_img2d = np.zeros_like(img2d)

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

            # do some processing here
            tile2d = func2d(tile2d, **kwargs)

            # place frame inside the new image
            new_img2d[tile.roi.x:tile.roi.x + tile.roi.w,
                      tile.roi.y:tile.roi.y + tile.roi.h] = tile2d

    if img2d.ndim != 2:
        raise tile_has_wrong_dimensionality(img2d.ndim)

    return new_img2d


def tile_has_wrong_dimensionality(num_dim: int) -> ValueError:
    """Check if the array as exactly 2 dimensions

    :param num_dim: number of dimensions
    :type num_dim: int
    :return: error message
    :rtype: ValueError
    """

    return ValueError(f"{str(num_dim)} does not equal 2.")
