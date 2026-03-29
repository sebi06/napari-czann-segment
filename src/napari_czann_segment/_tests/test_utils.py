# -*- coding: utf-8 -*-

import os
import tempfile
import pytest
from napari_czann_segment.utils import (
    file_not_found,
    check_file,
    get_fname_woext,
    Rectangle,
    get_rectangle_from_image,
    TileMethod,
    SupportedWindow,
    setup_log,
)


class TestFileNotFound:
    def test_returns_file_not_found_error(self):
        err = file_not_found("/some/fake/path.txt")
        assert isinstance(err, FileNotFoundError)
        assert "/some/fake/path.txt" in str(err)


class TestCheckFile:
    def test_valid_file(self, tmp_path):
        f = tmp_path / "exists.txt"
        f.write_text("hello")
        # Should not raise
        check_file(f)

    def test_invalid_file(self, tmp_path):
        f = tmp_path / "does_not_exist.txt"
        with pytest.raises(FileNotFoundError):
            check_file(f)

    def test_directory_is_not_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            check_file(tmp_path)


class TestGetFnameWoext:
    @pytest.mark.parametrize(
        "path, expected_name, expected_ext",
        [
            ("myfile.txt", "myfile", ".txt"),
            ("archive.tar.gz", "archive", ".tar.gz"),
            ("/path/to/model.onnx", "/path/to/model", ".onnx"),
            ("image.ome.tiff", "image", ".ome.tiff"),
            ("no_extension", "no_extension", ""),
        ],
    )
    def test_various_extensions(self, path, expected_name, expected_ext):
        name, ext = get_fname_woext(path)
        assert name == expected_name
        assert ext == expected_ext


class TestRectangle:
    def test_named_tuple_fields(self):
        r = Rectangle(x=10, y=20, w=100, h=200)
        assert r.x == 10
        assert r.y == 20
        assert r.w == 100
        assert r.h == 200

    def test_get_rectangle_from_image(self):
        r = get_rectangle_from_image(5, 10, 640, 480)
        assert isinstance(r, Rectangle)
        assert r == Rectangle(x=5, y=10, w=640, h=480)


class TestEnums:
    def test_tile_method_values(self):
        assert TileMethod.CZTILE.value == 1
        assert TileMethod.TILER.value == 2
        assert TileMethod.RYOMEN.value == 3

    def test_supported_window_has_all_members(self):
        expected = {
            "boxcar",
            "triang",
            "blackman",
            "hamming",
            "hann",
            "bartlett",
            "parzen",
            "bohman",
            "blackmanharris",
            "nuttall",
            "barthann",
            "overlaptile",
            "none",
        }
        assert set(SupportedWindow.__members__.keys()) == expected


class TestSetupLog:
    def test_returns_logger(self):
        log = setup_log("test_utils_logger")
        assert log.name == "test_utils_logger"

    def test_logfile_creation(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        log = setup_log("file_logger", create_logfile=True)
        logfile = tmp_path / "test_file_logger.log"
        assert logfile.exists()
