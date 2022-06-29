# -*- coding: utf-8 -*-

#################################################################
# File        : demo_tiling.py
# Author      : sebi06, Team Enchilada
#
# Disclaimer: This code is purely experimental. Feel free to
# use it at your own risk.
#
#################################################################

import numpy as np
import matplotlib.pyplot as plt
from cztile.fixed_total_area_strategy import AlmostEqualBorderFixedTotalAreaStrategy2D
from typing import Union
from aicsimageio import AICSImage
import dask.array as da
from skimage.filters import gaussian
from napari_czann_segment import get_testdata, utils
from tqdm import tqdm


def process2d(image2d: Union[np.ndarray, da.Array], **kwargs: int) -> Union[np.ndarray, da.Array]:
    # insert or modify the desired processing function here
    image2d = gaussian(image2d, sigma=kwargs["sigma"],
                       preserve_range=True,
                       mode='nearest').astype(image2d.dtype)

    return image2d


###########################################################

# get data to test the functionality
img_name = "PGC_20X.ome.tiff"
img_path = get_testdata.get_imagefile(img_name)
print("Try to read file: ", img_path)

# read the CZI using AICSImageIO and aicspylibczi
aics_img = AICSImage(img_path)
print("Dimension Original Image:", aics_img.dims)
print("Array Shape Original Image:", aics_img.shape)

# read the image data as numpy or dask array
img2d = aics_img.get_image_data("YX", C=0, T=0, Z=0)
# img = aics_img.get_image_dask_data()

new_img2d = da.zeros(shape=(aics_img.dims.Y, aics_img.dims.X),
                     chunks=(aics_img.dims.Y, aics_img.dims.X))
# new_img2d = da.zeros_like(img2d, chunks=(img2d.shape[0], img2d.shape[1], img2d.shape[2]))

# create a "tile" by specifying the desired tile dimension and the
# minimum required overlap between tiles (depends on the processing)
tiler = AlmostEqualBorderFixedTotalAreaStrategy2D(total_tile_width=1600,
                                                  total_tile_height=1400,
                                                  min_border_width=128)

# get the size of the bounding rectangle for the scene
rt = utils.get_rectangle_from_image(x=0,
                                    y=0,
                                    sizex=aics_img.dims.X,
                                    sizey=aics_img.dims.Y)
# create tiler object
tiles = tiler.tile_rectangle(rt)

# show the created tile locations
for tile in tiles:
    print(tile.roi.x, tile.roi.y, tile.roi.w, tile.roi.h)

# loop over all tiles
for tile in tqdm(tiles):
    # get a single frame based on the roi -be aware of the typical y - x - h - w topic when using array
    tile2d = img2d[tile.roi.y:tile.roi.y + tile.roi.h, tile.roi.x:tile.roi.x + tile.roi.w]

    # do some processing here
    tile2d = process2d(tile2d, sigma=3)

    # place frame inside the new image
    new_img2d[tile.roi.y:tile.roi.y + tile.roi.h, tile.roi.x:tile.roi.x + tile.roi.w] = tile2d

# show the results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
ax1.imshow(img2d, interpolation="nearest", cmap="gray", vmin=32000, vmax=35000)
ax1.set_title('Original')
ax2.imshow(new_img2d, interpolation="nearest", cmap="gray", vmin=32000, vmax=35000)
ax2.set_title('Processed - TileWise')
plt.show()

print("Done.")
