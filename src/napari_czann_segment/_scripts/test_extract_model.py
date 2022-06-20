# -*- coding: utf-8 -*-

#################################################################
# File        : test_extract_model.py
# Author      : sebi06
#
# Disclaimer: This code is purely experimental. Feel free to
# use it at your own risk.
#
#################################################################

from napari_czann_segment import get_testdata
from pathlib import Path
import tempfile
import os
from czmodel.convert import DefaultConverter
from pathlib import Path

# get data to test the functionality
name_czann = "PGC_20X_nucleus_detector.czann"

czann_file = get_testdata.get_modelfile(name_czann)

target_dir = os.path.join(os.getcwd(), "_tmp")

# this is the old way to do it
with tempfile.TemporaryDirectory() as temp_path:
    # this is the new way of unpacking using the czann files
    model_metadata, model_path = DefaultConverter().unpack_model(model_file=czann_file, target_dir=Path(temp_path))

    # show model metadata
    print(model_metadata, "\n")

    # read individual model metadata
    print(model_metadata.model_type, "\n")

    # get model metadata as dictionary
    model_metadata_dict = model_metadata._asdict()

    for k, v in model_metadata_dict.items():
        print(k, "=", v)

print("\nDone.")
