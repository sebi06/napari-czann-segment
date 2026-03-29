# -*- coding: utf-8 -*-

import os
import pytest
from napari_czann_segment import get_testdata


class TestGetModelfile:
    def test_known_model(self):
        path = get_testdata.get_modelfile("simple_regmodel.czann")
        assert os.path.isfile(path)
        assert path.endswith("simple_regmodel.czann")

    def test_missing_model_raises(self):
        with pytest.raises(FileNotFoundError):
            get_testdata.get_modelfile("nonexistent_model.czann")


class TestGetImagefile:
    def test_known_image(self):
        path = get_testdata.get_imagefile("PGC_20X.czi")
        assert os.path.isfile(path)

    def test_missing_image_raises(self):
        with pytest.raises(FileNotFoundError):
            get_testdata.get_imagefile("nonexistent_image.tiff")
