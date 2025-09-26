# -*- coding: utf-8 -*-

#################################################################
# File        : get_testdata.py
# Author      : sebi06, Team Enchilada
#
# Disclaimer: This code is purely experimental. Feel free to
# use it at your own risk.
#
#################################################################

import os

try:
    from importlib.resources import files
except ImportError:
    # Fallback for Python < 3.9
    from importlib_resources import files
from napari_czann_segment import utils

PACKAGE_NAME = r"napari_czann_segment"
PACKAGE_DATA = r"_data/"


def get_modelfile(name_czann: str = "PGC_nucleus_detector.czann") -> str:
    """Function to get the path of a CZANN for testing

    Args:
        name_czann (str, optional): Name of the CZANN model. Defaults to "PGC_nucleus_detector.czann".

    Returns:
        str: Absolute path of the CZANN (or CZMODEL) file
    """

    # Get the data directory using modern importlib.resources
    package_files = files(PACKAGE_NAME)
    datadir = str(package_files / PACKAGE_DATA.strip("/"))
    czann_file = os.path.join(datadir, name_czann)

    utils.check_file(czann_file)

    return czann_file


def get_imagefile(name_imagefile: str = "PGC_10x_S02.czi") -> str:
    """Function to get the path of a *.czi or *.ome.tiff for testing. Lower and upper case are important.

    Args:
        name_imagefile (str, optional): Name of the file. Defaults to "PGC_10x_S02.czi".

    Returns:
        str: Absolute path of the file
    """

    # Get the data directory using modern importlib.resources
    package_files = files(PACKAGE_NAME)
    datadir = str(package_files / PACKAGE_DATA.strip("/"))
    path_imagefile = os.path.join(datadir, name_imagefile)

    utils.check_file(path_imagefile)

    return path_imagefile
