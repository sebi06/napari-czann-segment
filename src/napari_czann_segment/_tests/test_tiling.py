from bioio import BioImage
from cztile.fixed_total_area_strategy_2d import AlmostEqualBorderFixedTotalAreaStrategy2D
from cztile.tiling_strategy import Region2D, TileInput
import numpy as np
import pytest
from napari_czann_segment.dock_widget import setup_log
from napari_czann_segment import get_testdata


logger = setup_log("Napari-CZANN")


@pytest.mark.parametrize(
    "image, total_tile_length, min_border_length",
    [
        ("PGC_20X.ome.tiff", 1024, 128),
    ],
)
def test_tiling2d(image: str, total_tile_length: int, min_border_length: int) -> None:

    image_file = get_testdata.get_imagefile(image)

    bioio_img = BioImage(image_file)
    logger.info(f"Dimension Original Image: {bioio_img.dims}")
    logger.info(f"Array Shape Original Image: {bioio_img.shape}")

    # read the image data as numpy or dask array
    img = bioio_img.get_image_data()

    # get 2d array
    img2d = np.squeeze(img)  # shape (2755, 3675)

    # minimum required overlap between tiles (depends on the processing)
    tiler = AlmostEqualBorderFixedTotalAreaStrategy2D(
        width=TileInput(total_tile_length=total_tile_length, min_border_length=min_border_length),
        height=TileInput(total_tile_length=total_tile_length, min_border_length=min_border_length),
    )

    # create the tiles --> region2d = (x, y, w, h)
    region2d = Region2D(x=0, y=0, w=img2d.shape[1], h=img2d.shape[0])
    tiles = tiler.calculate_2d_tiles(region2d=region2d)

    # show the tile locations
    for tile in tiles:

        # get a single frame based on the tile coordinates and size
        tile2d = img2d[tile.roi.y : tile.roi.y + tile.roi.h, tile.roi.x : tile.roi.x + tile.roi.w]

        assert tile2d is not None
        assert tile.roi.w <= total_tile_length
        assert tile.roi.h <= total_tile_length
        assert tile2d.shape[0] <= total_tile_length
        assert tile2d.shape[1] <= total_tile_length
