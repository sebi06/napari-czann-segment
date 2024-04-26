# -*- coding: utf-8 -*-

#################################################################
# File        : utils.py
# Author      : sebi06, Team Enchilada
#
# Disclaimer: This code is purely experimental. Feel free to
# use it at your own risk.
#
#################################################################

from typing import List, NamedTuple, Union, Tuple, Callable, NamedTuple
import os
from pathlib import Path
from typing import NamedTuple
import logging
import configparser
import datetime
from enum import Enum


def file_not_found(file_path: os.PathLike) -> FileNotFoundError:
    """Raise exception in case a file does not exist

    :param file_path: file path that does not exist
    :type file_path: os.PathLike
    :return: error message
    :rtype: FileNotFoundError
    """

    return FileNotFoundError(f"{file_path} could not be found.")


def check_file(file_path: os.PathLike) -> None:
    """Check

    :param file_path: [description]
    :type file_path: os.PathLike
    :raises file_not_found: [description]
    """

    # check if this is a file
    if os.path.isfile(file_path):
        print("Valid file: ", file_path, "found.")
    else:
        raise file_not_found(file_path)


def get_fname_woext(file_path: str) -> Tuple[str, str]:
    """Get the complete path of a file without the extension
    It also will works for extensions like myfile.abc.xyz
    The output will be: myfile

    :param filepath: complete fiepath
    :type filepath: str
    :return: complete filepath without extension
    :rtype: str
    """
    # create empty string
    real_extension = ""

    # get all part of the file extension
    sufs = Path(file_path).suffixes
    for s in sufs:
        real_extension = real_extension + s

    # remove real extension from filepath
    filepath_woext = file_path.replace(real_extension, "")

    return filepath_woext, real_extension


class Rectangle(NamedTuple):
    """A class representing a rectangle with x, y coordinates and width, height dimensions."""

    x: int
    y: int
    w: int
    h: int


def get_rectangle_from_image(
    x: int, y: int, sizex: int, sizey: int
) -> Rectangle:
    """
    Create a rectangle object from the given coordinates and sizes.

    Parameters:
    - x (int): The x-coordinate of the top-left corner of the rectangle.
    - y (int): The y-coordinate of the top-left corner of the rectangle.
    - sizex (int): The width of the rectangle.
    - sizey (int): The height of the rectangle.

    Returns:
    - Rectangle: A rectangle object with the specified coordinates and sizes.
    """
    rt = Rectangle(x=x, y=y, w=sizex, h=sizey)

    return rt


from enum import Enum


class TileMethod(Enum):
    """
    Enumeration class representing different tiling methods.

    Attributes:
        CZTILE (int): Represents the CZTILE method.
        TILER (int): Represents the TILER method.
        RYOMEN (int): Represents the RYOMEN method.
    """

    CZTILE = 1
    TILER = 2
    RYOMEN = 3


from enum import Enum


class SupportedWindow(Enum):
    """
    Enum class representing supported window types merging tiles for the Tiler package.
    """

    boxcar = 1
    triang = 2
    blackman = 3
    hamming = 4
    hann = 5
    bartlett = 6
    parzen = 7
    bohman = 8
    blackmanharris = 9
    nuttall = 10
    barthann = 11
    overlaptile = 12
    none = 13
