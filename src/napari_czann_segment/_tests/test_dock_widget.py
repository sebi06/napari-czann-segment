import dask.array as da
import pytest
from napari.layers import Image, Labels


@pytest.fixture
def im_layer():
    """
    Fixture that returns an Image layer with random data.

    Returns:
        napari.layers.Image: Image layer with random data.
    """
    return Image(da.random.random((5, 100, 100)), name="im")


@pytest.fixture
def labels_layer():
    """
    Fixture function that returns a Labels layer with random integer values.

    Returns:
        Labels: A Labels layer with random integer values.
    """
    return Labels(da.random.randint(0, 255, (5, 100, 100)), name="lab")
