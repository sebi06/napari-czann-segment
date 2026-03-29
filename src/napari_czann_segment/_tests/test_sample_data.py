# -*- coding: utf-8 -*-

import numpy as np
from napari_czann_segment._sample_data import make_sample_data


def test_make_sample_data():
    data = make_sample_data()
    assert isinstance(data, np.ndarray)
    assert data.shape == (512, 512)
    assert data.dtype == np.float64
    assert data.min() >= 0.0
    assert data.max() <= 1.0
