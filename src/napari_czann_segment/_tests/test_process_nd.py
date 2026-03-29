# -*- coding: utf-8 -*-

import numpy as np
import pytest
from napari_czann_segment.process_nd import label_nd


class TestLabelNd:
    def test_2d_single_class(self):
        """A simple 2D segmentation with one label value should produce labelled regions."""
        seg = np.array(
            [
                [0, 0, 1, 1],
                [0, 0, 1, 1],
                [2, 2, 0, 0],
                [2, 2, 0, 0],
            ]
        )
        # Wrap in 5-D (T, C, Z, Y, X) to match expected input
        seg5d = seg[np.newaxis, np.newaxis, np.newaxis, :, :]
        labels = label_nd(seg5d, labelvalue=1)
        assert labels.shape == seg5d.shape

        labels_np = np.asarray(labels)
        # Pixels where seg == 1 should have a label > 0
        assert np.all(labels_np[0, 0, 0, 0:2, 2:4] > 0)
        # Pixels where seg != 1 should remain 0
        assert np.all(labels_np[0, 0, 0, 2:4, :] == 0)
        assert np.all(labels_np[0, 0, 0, 0:2, 0:2] == 0)

    def test_no_matching_pixels(self):
        """If labelvalue does not appear, all labels should be zero."""
        seg = np.zeros((1, 1, 1, 8, 8), dtype=int)
        labels = label_nd(seg, labelvalue=5)
        assert np.all(np.asarray(labels) == 0)

    def test_multiple_connected_components(self):
        """Two separate regions of the same label should get distinct labels."""
        seg = np.zeros((1, 1, 1, 6, 6), dtype=int)
        seg[0, 0, 0, 0:2, 0:2] = 3  # region A
        seg[0, 0, 0, 4:6, 4:6] = 3  # region B
        labels = label_nd(seg, labelvalue=3)
        labels_2d = np.asarray(labels)[0, 0, 0]
        unique = set(labels_2d[labels_2d > 0])
        assert len(unique) == 2  # two distinct components

    def test_multi_frame(self):
        """label_nd should iterate over leading dimensions correctly."""
        seg = np.zeros((2, 1, 1, 4, 4), dtype=int)
        seg[0, 0, 0, 0:2, 0:2] = 1
        seg[1, 0, 0, 2:4, 2:4] = 1
        labels = label_nd(seg, labelvalue=1)
        labels_np = np.asarray(labels)
        assert labels_np.shape == seg.shape
        # Frame 0: top-left labelled
        assert np.all(labels_np[0, 0, 0, 0:2, 0:2] > 0)
        assert np.all(labels_np[0, 0, 0, 2:4, 2:4] == 0)
        # Frame 1: bottom-right labelled
        assert np.all(labels_np[1, 0, 0, 2:4, 2:4] > 0)
        assert np.all(labels_np[1, 0, 0, 0:2, 0:2] == 0)
