import dask.array as da
import pytest
from napari.layers import Image, Labels
from qtpy.QtCore import Qt


@pytest.fixture
def im_layer():
    return Image(da.random.random((5, 100, 100)), name="im")


@pytest.fixture
def labels_layer():
    return Labels(da.random.randint(0, 255, (5, 100, 100)), name="lab")
