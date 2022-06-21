# -*- coding: utf-8 -*-

#################################################################
# File        : ndarray_with_2d_tiling.py
# Author      : sebi06, Team Enchilada
#
# Disclaimer: This code is purely experimental. Feel free to
# use it at your own risk.
#
#################################################################

import itertools
from aicsimageio import AICSImage
import numpy as np
from napari_czann_segment import get_testdata, tiling
from typing import Union
import dask.array as da
import napari
import napari.types


def _process2d(tile2d: Union[np.ndarray, da.Array], **kwargs: int) -> Union[np.ndarray, da.Array]:
    """example for a 2d processing function tp be applied to an individual tile

    :param tile2d: the 2d tile image to be processed
    :type tile2d: Union[np.ndarray, da.array]
    :return: [description]
    :rtype: Union[np.ndarray, da.array]
    """

    if tile2d.ndim == 2:
        from skimage.filters import gaussian

        # insert or modify the desired processing function here
        tile2d = gaussian(tile2d, sigma=kwargs["sigma"],
                          preserve_range=True,
                          mode='nearest').astype(tile2d.dtype)
    if tile2d.ndim != 2:
        raise tiling.tile_has_wrong_dimensionality(tile2d.ndim)

    return tile2d


####################################################################################


# get data to test the functionality
img_name = "2x2_T=2_Z=3_CH=1.ome.tiff"
img_path = get_testdata.get_imagefile(img_name)

# read the CZI using AICSImageIO and aicspylibczi
aics_img = AICSImage(img_path)
print("Dimension Original Image:", aics_img.dims)
print("Array Shape Original Image:", aics_img.shape)

# get the image data
img = aics_img.get_image_dask_data()
new_img = da.zeros_like(img, chunks=img.shape)

# get the shape without the XY dimensions
shape = aics_img.dims.shape[:-2]

# create the "values" each for-loop iterates over
loopover = [range(s) for s in shape]
prod = itertools.product(*loopover)

for idx in prod:

    # create list of slice objects based on the shape
    # sl = len(shape) * [slice(None)]
    sl = len(shape) * [np.s_[0:1]]

    # insert the correct index into the respective slice objects for all dimensions
    for nd in range(len(shape)):
        # sl[nd] = slice(idx[nd])
        sl[nd] = idx[nd]

    # extract the 2D image from the n-dims stack using the list of slice objects
    img2d = np.squeeze(img[tuple(sl)])

    # process the whole 2d image - make sure to use the correct **kwargs
    new_img2d = tiling.process2d_tiles(_process2d,
                                       img2d,
                                       tile_width=1000,
                                       tile_height=800,
                                       min_border_width=128,
                                       use_dask=False,
                                       sigma=11)

    # insert new 2D after tile-wise processing into nd array
    new_img[tuple(sl)] = new_img2d

# add the image as a layer to the napari viewer
viewer = napari.Viewer()
nd_layer = viewer.add_image(new_img, name="processed",
                            blending="translucent")

print("Done.")

napari.run()
