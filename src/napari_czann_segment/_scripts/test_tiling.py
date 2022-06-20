# -*- coding: utf-8 -*-

#################################################################
# File        : test_tiling.py
# Author      : sebi06
#
# Disclaimer: This code is purely experimental. Feel free to
# use it at your own risk.
#
#################################################################

import numpy as np
import matplotlib.pyplot as plt
from cztile.fixed_total_area_strategy import AlmostEqualBorderFixedTotalAreaStrategy2D
import os
from typing import Union
from pylibCZIrw import czi as pyczi
import dask.array as da
from skimage.filters import gaussian
from napari_czann_segment import get_testdata
from tqdm import tqdm


def process2d(image2d: Union[np.ndarray, da.Array], **kwargs: int) -> Union[np.ndarray, da.Array]:
    # insert or modify the desired processing function here
    image2d = gaussian(image2d, sigma=kwargs["sigma"],
                       preserve_range=True,
                       mode='nearest').astype(image2d.dtype)

    return image2d


###########################################################

# get data to test the functionality
name_czi = "PGC_10x_S02.czi"
czi_file = get_testdata.get_czifile(name_czi)
print("Try to read CZI file: ", czi_file)

# check if this is a file
if os.path.isfile(czi_file):
    print("Valid file found.")
else:
    print("File not exist")
    os._exit(0)

with pyczi.open_czi(czi_file) as czidoc:
    # get the image dimensions as an dictionary, where the key identifies the dimension
    total_bounding_box = czidoc.total_bounding_box

    # get the bounding boxes for each individual scene
    scenes_bounding_rectangle = czidoc.scenes_bounding_rectangle

    # read the actual image into an array
    img2d = czidoc.read(plane={'C': 0, 'Z': 0, 'T': 0})[..., 0]

# create a new array to hold the processed image
sizeX = scenes_bounding_rectangle[0].w
sizeY = scenes_bounding_rectangle[0].h

new_img2d = da.zeros(shape=(sizeY, sizeX), chunks=(sizeY, sizeX))
# new_img2d = da.zeros_like(img2d, chunks=(img2d.shape[0], img2d.shape[1], img2d.shape[2]))

# create a "tile" by specifying the desired tile dimension and the
# minimum required overlap between tiles (depends on the processing)
tiler = AlmostEqualBorderFixedTotalAreaStrategy2D(total_tile_width=1600,
                                                  total_tile_height=1400,
                                                  min_border_width=128)

# get the size of the bounding rectangle for the scene
tiles = tiler.tile_rectangle(scenes_bounding_rectangle[0])

# show the created tile locations
for tile in tiles:
    print(tile.roi.x, tile.roi.y, tile.roi.w, tile.roi.h)

# loop over all tiles
for tile in tqdm(tiles):
    # get a single frame based on the roi -be aware of the typical y - x - h - w topic when using array
    tile2d = img2d[tile.roi.y:tile.roi.y + tile.roi.h, tile.roi.x:tile.roi.x + tile.roi.w]

    # do some processing here
    tile2d = process2d(tile2d, sigma=7)

    # place frame inside the new image
    new_img2d[tile.roi.y:tile.roi.y + tile.roi.h, tile.roi.x:tile.roi.x + tile.roi.w] = tile2d

# show the results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
ax1.imshow(img2d, interpolation="nearest", cmap="gray")
ax1.set_title('Original')
ax2.imshow(new_img2d, interpolation="nearest", cmap="gray")
ax2.set_title('Processed - TileWise')
plt.show()

print("Done.")
