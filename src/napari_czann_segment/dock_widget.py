# -*- coding: utf-8 -*-

#################################################################
# File        : dock_widget.py
# Author      : sebi06, Team Enchilada
#
# Disclaimer: This code is purely experimental. Feel free to
# use it at your own risk.
#
# Partially based upon the workshop-demo plugin: https://github.com/DragaDoncila/workshop-demo
# and the instructions and example found here: https://napari.org/plugins/stable/for_plugin_developers.html
#
#################################################################


import numpy as np
from napari.layers import Labels, Image
from .process_nd import label_nd
from .predict import predict_ndarray
import tempfile
from pathlib import Path
from czmodel.convert import DefaultConverter
from typing import Dict, List, Tuple, Union
from qtpy.QtWidgets import (QComboBox, QHBoxLayout, QLabel, QPushButton, QLineEdit, QListWidgetItem,
                            QVBoxLayout, QWidget, QFileDialog, QDialogButtonBox, QSlider)

from qtpy.QtCore import Qt, Signal, QObject, QEvent
from magicgui.widgets import FileEdit, PushButton, Slider, Container, Label, CheckBox
from magicgui.types import FileDialogMode


# our manifest widget command points to this class
class segment_with_czann(QWidget):
    """Widget allows selection of an Image layer, a model file and the desired border width
    and returns as many new layers as the segmentation model has classes

    Important: Segmentation is done Slice-by-Slice.

    """

    def __init__(self, napari_viewer):
        """Initialize widget
        Parameters
        ----------
        napari_viewer : napari.utils._proxies.PublicOnlyProxy
            public proxy for the napari viewer object
        """
        super().__init__()
        self.viewer = napari_viewer

        # set default values
        self.min_overlap: int = 128
        self.model_metadata = None
        self.czann_file: str = ""
        self.dnn_tile_width: int = 1024
        self.dnn_tile_height: int = 1024

        # TODO : Enable GPU support and make it work
        self.use_gpu: bool = False

        # create a layout
        self.setLayout(QVBoxLayout())

        # add a label
        self.layout().addWidget(QLabel("Model File Selection"))

        # define filter to allowed file extensions
        model_extension = "*.czann"

        # create the FileEdit widget and add to the layout and connect it
        self.filename_edit = FileEdit(mode=FileDialogMode.EXISTING_FILE,
                                      value="",
                                      filter=model_extension)

        self.layout().addWidget(self.filename_edit.native)
        self.filename_edit.line_edit.changed.connect(self._file_changed)

        # add button for reading the model metadata
        self.read_modeldata_btn = QPushButton("Reload minimum overlap from Metadata")
        self.read_modeldata_btn.clicked.connect(self._read_min_overlap)
        self.layout().addWidget(self.read_modeldata_btn)

        # add label and slider for adjusting the minimum overlap
        min_overlap_label = QLabel("Adjust minimum overlap for Segmentation")
        self.min_overlap_slider = Slider(orientation="horizontal",
                                         label="Minimum Overlap",
                                         value=128,
                                         min=8,
                                         max=256,
                                         step=1,
                                         readout=True,
                                         tooltip="Adjust the desired minimum overlap",
                                         tracking=False)

        self.layout().addWidget(min_overlap_label)
        self.layout().addWidget(self.min_overlap_slider.native)
        self.min_overlap_slider.changed.connect(self._update_min_overlap)

        # make a combo box for selecting the image layers
        self.layer_combos = []
        self.image_layer_combo = self.add_image_combo_box("Image Layer")

        # if the user adds or removes layers - update the combo box
        self.viewer.layers.events.inserted.connect(self._reset_layer_options)
        self.viewer.layers.events.removed.connect(self._reset_layer_options)

        # TODO : This checkbox is currentyl hidden
        # add the checkbox the use the GPU for the inference
        self.use_gpu_checkbox = CheckBox(name="Use GPU for inference",
                                         visible=False,
                                         enabled=False,
                                         value=self.use_gpu)
        self.use_gpu_checkbox.clicked.connect(self._use_gpu_changed)
        self.layout().addWidget(self.use_gpu_checkbox.native)

        # make button for reading the model metadata
        self.segment_btn = QPushButton("Segment selected Image Layer")
        self.segment_btn.clicked.connect(self._segment)
        self.layout().addWidget(self.segment_btn)

    def add_image_combo_box(self, label_text):
        """Add combo box with the given image layers as items
        Parameters
        ----------
        label_text : str
            Text to add next to combo box
        Returns
        -------
        QComboBox
            Combo box with labels layers as items
        """

        # make a new row to put label and combo box in
        combo_row = QWidget()
        combo_row.setLayout(QHBoxLayout())

        # we don't want margins so it looks all tidy
        combo_row.layout().setContentsMargins(0, 0, 0, 0)

        new_combo_label = QLabel(label_text)
        combo_row.layout().addWidget(new_combo_label)

        new_layer_combo = QComboBox(self)
        # only adding labels layers
        new_layer_combo.addItems(
            [layer.name for layer in self.viewer.layers if isinstance(layer, Image)]
        )
        combo_row.layout().addWidget(new_layer_combo)

        # saving to a list so we can iterate through all combo boxes to reset choices
        self.layer_combos.append(new_layer_combo)
        self.layout().addWidget(combo_row)

        # returning the combo box so we know which is which when we click run
        return new_layer_combo

    def _read_min_overlap(self):
        """Get model minimum overlap and store them"""

        # extract the model information and path
        with tempfile.TemporaryDirectory() as temp_path:
            self.model_metadata, self.model_path = DefaultConverter().unpack_model(model_file=self.czann_file,
                                                                                   target_dir=Path(temp_path))

        # update the value for the minimum overlap and the respective slider
        self.min_overlap = self.model_metadata.min_overlap[0]
        self.min_overlap_slider.value = self.min_overlap
        print("Updated value for minimum overlap and Slider from ModelMetaData: ", self.min_overlap)

    def _segment(self):

        print("Run the segmentation.")

        # deactivate the button
        self.segment_btn.isEnabled = False

        # grab the layer using the combo box item text as the layer name
        img_layer = self.viewer.layers[self.image_layer_combo.currentText()]

        print("CZANN Modelfile:", self.czann_file)
        print("Minimum Tile Overlap:", self.min_overlap)
        print("Use GPU acceleration:", self.use_gpu)

        modeldata, seg_complete = predict_ndarray(self.czann_file,
                                                  img_layer.data,
                                                  border=self.min_overlap,
                                                  use_gpu=self.use_gpu)

        # create a list of label values
        label_values = list(range(1, len(modeldata.classes) + 1))

        # get individual outputs for all classes from the label image
        for c in range(len(modeldata.classes)):
            # get the pixels for which the value is equal to current class value
            print("Class Name:", modeldata.classes[c], "Prediction Pixel Value:", c)

            # get all pixels with a specific value as boolean array, convert to numpy array and label
            labels_current_class = label_nd(seg_complete,
                                            labelvalue=label_values[c])

            # add new image layer
            self.viewer.add_labels(labels_current_class,
                                   name=f"{img_layer.name}_" + modeldata.classes[c],
                                   num_colors=256,
                                   scale=img_layer.scale,
                                   opacity=0.7,
                                   blending="translucent")

        # reactivate the button
        self.segment_btn.isEnabled = True

    def _update_min_overlap(self):

        # update in case the slider was adjusted
        self.min_overlap = self.min_overlap_slider.value
        self._check_min_overlap()
        print("New minimum overlap value: ", self.min_overlap)

    def _file_changed(self):

        self.czann_file = str(self.filename_edit.value.absolute()
                              ).replace("\\", "/").replace("//", "/")
        print("Model Path: ", self.czann_file)

        # update the minimum overlap
        self._read_min_overlap()

    def _reset_layer_options(self, event):
        """Clear existing combo boxes and repopulate
        Parameters
        ----------
        event : event
            Clear existing combo box items and query viewer for all image layers
        """
        for combo in self.layer_combos:
            combo.clear()
            combo.addItems(
                [
                    layer.name
                    for layer in self.viewer.layers
                    if isinstance(layer, Image)
                ]
            )

    def _check_min_overlap(self):

        # Minimum border width must be less than half the tile size.
        min_tilesize = min(self.dnn_tile_width, self.dnn_tile_width)

        # check
        if self.min_overlap >= np.round(min_tilesize / 2, 0):
            self.min_overlap = int(np.round(min_tilesize / 2, 0) - 1)
            print("Minimum border width must be less than half the tile size.")
            print("Adjusted minimum overlap :", self.min_overlap)
            self.min_overlap_slider.value = self.min_overlap

    def _use_gpu_changed(self):

        if self.use_gpu_checkbox.value:
            self.use_gpu = True
            print("Use GPU for inference.")
        if not self.seglabel_dask_checkbox.value:
            self.use_gpu = False
            print("Use CPU for inference.")
