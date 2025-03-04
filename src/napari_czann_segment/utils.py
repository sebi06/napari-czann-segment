# -*- coding: utf-8 -*-

#################################################################
# File        : utils.py
# Author      : sebi06, Team Enchilada
#
# Disclaimer: This code is purely experimental. Feel free to
# use it at your own risk.
#
#################################################################

from typing import List, NamedTuple, Union, Tuple, Callable
import os
from pathlib import Path


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
    real_extension = ''

    # get all part of the file extension
    sufs = Path(file_path).suffixes
    for s in sufs:
        real_extension = real_extension + s

    # remove real extension from filepath
    filepath_woext = file_path.replace(real_extension, '')

    return filepath_woext, real_extension


def get_rectangle_from_image(x: int, y: int, sizex: int, sizey: int) -> NamedTuple("Rectangle", [("x", int), ("y", int), ("w", int), ("h", int)]):

    Rectangle = NamedTuple("Rectangle", [("x", int), ("y", int), ("w", int), ("h", int)])
    rt = Rectangle(x=x,
                   y=y,
                   w=sizex,
                   h=sizey)

    return rt
