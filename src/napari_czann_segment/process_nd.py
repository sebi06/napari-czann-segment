# -*- coding: utf-8 -*-

#################################################################
# File        : process_nd.py
# Author      : sebi06, Team Enchilada
#
# Disclaimer: This code is purely experimental. Feel free to
# use it at your own risk.
#
#################################################################

import dask.array as da
from skimage.measure import label
import numpy as np
import itertools
from typing import List, NamedTuple, Union, Tuple, Callable


def label_nd(seg: Union[np.ndarray, da.Array],
             labelvalue: int = 0,
             # output_dask: bool = False,
             ) -> Union[np.ndarray, da.Array]:

    shape_woXY = seg.shape[:-2]

    label_complete = da.zeros_like(seg, chunks=seg.shape)

    # create the "values" each for-loop iterates over
    loopover = [range(s) for s in shape_woXY]
    prod = itertools.product(*loopover)

    # loop over all dimensions
    for idx in prod:

        # create list of slice objects based on the shape
        sl = len(shape_woXY) * [np.s_[0:1]]

        # insert the correct index into the respective slice objects for all dimensions
        for nd in range(len(shape_woXY)):
            sl[nd] = idx[nd]

        # extract the 2D image from the n-dims stack using the list of slice objects
        seg2d = np.squeeze(seg[tuple(sl)])

        # process the whole 2d image - make sure to use the correct **kwargs
        new_label2d = label((seg2d == labelvalue).astype(float))

        # insert new 2D after tile-wise processing into nd array
        label_complete[tuple(sl)] = new_label2d

        print("Datatype Labels", type(label_complete))

    return label_complete
