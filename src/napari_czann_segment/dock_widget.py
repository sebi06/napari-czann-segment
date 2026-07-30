# -*- coding: utf-8 -*-

#################################################################
# File        : dock_widget.py
# Author      : sebi06, Team Enchilada
#
# Disclaimer: This code is purely experimental. Feel free to
# use it at your own risk.
#
#
# Remarks: Requires czmodel >= 5.0
#################################################################

import numpy as np
import napari
from napari.layers import Image
from napari_czann_segment.process_nd import label_nd
from napari_czann_segment.predict import predict_ndarray
from napari_czann_segment.utils import TileMethod, SupportedWindow
from napari_czann_segment.onnx_inference import is_gpu_available
import tempfile
from pathlib import Path

# from czmodel.pytorch.convert import DefaultConverter
from czmodel.core.util._extract_model import extract_czann_model
from czmodel import ModelType
from napari_czann_segment.czseg_parser import extract_czseg_model
from typing import Dict
from qtpy.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QTableWidget,
    QTableWidgetItem,
)

from qtpy.QtCore import Qt
from qtpy.QtGui import QFont
from magicgui.widgets import FileEdit, Slider, CheckBox, PushButton, ComboBox
from magicgui.types import FileDialogMode
import warnings
from .utils import setup_log


def _resolve_do_rescale(value):
    """Resolve missing model scaling metadata to the legacy default."""
    return True if value is None else bool(value)


class TableWidget(QWidget):
    """
    A custom widget that displays a table with parameter-value pairs.

    This widget provides methods to update the table with metadata entries and
    update the style of the table.

    Attributes:
        layout (QVBoxLayout): The layout of the widget.
        model_table (QTableWidget): The table widget that displays the parameter-value pairs.
    """

    def __init__(self) -> None:
        """
        Initialize the DockWidget.

        This method sets up the layout and initializes the QTableWidget.

        Parameters:
        None

        Returns:
        None
        """
        super().__init__()

        self.layout = QVBoxLayout(self)
        self.model_table = QTableWidget()

        self.model_table.setShowGrid(True)
        self.model_table.setHorizontalHeaderLabels(["Parameter", "Value"])
        header = self.model_table.horizontalHeader()
        header.setDefaultAlignment(Qt.AlignLeft)
        self.layout.addWidget(self.model_table)

    def update_model_metadata(self, md_dict: Dict) -> None:
        """
        Update the table with metadata entries.

        Args:
            md_dict (Dict): A dictionary containing the metadata entries.

        Returns:
            None
        """
        row_count = len(md_dict)
        col_count = 2
        self.model_table.setColumnCount(col_count)
        self.model_table.setRowCount(row_count)

        row = 0

        for key, value in md_dict.items():
            newkey = QTableWidgetItem(key)
            self.model_table.setItem(row, 0, newkey)
            newvalue = QTableWidgetItem(str(value))
            self.model_table.setItem(row, 1, newvalue)
            row += 1

        self.model_table.resizeColumnsToContents()
        self.model_table.resizeRowsToContents()
        self.model_table.adjustSize()

    def update_style(self) -> None:
        """
        Update the style of the table.

        This method sets the font size, type, and color of the header items.

        Returns:
            None
        """
        fnt = QFont()
        fnt.setPointSize(8)
        fnt.setBold(True)
        fnt.setFamily("Arial")

        # fc = (25, 25, 25)

        item1 = QTableWidgetItem("Parameter")
        item1.setFont(fnt)
        self.model_table.setHorizontalHeaderItem(0, item1)

        item2 = QTableWidgetItem("Value")
        item2.setFont(fnt)
        self.model_table.setHorizontalHeaderItem(1, item2)


# our manifest widget command points to this class
class segment_with_czann(QWidget):
    """Widget allows selection of an Image layer, a model file
    and the desired border width and returns as many new layers
    as the segmentation model has classes.
    For a regression model is return the processed image. Currently only one
    output channel is supported for regression models aka "Image-2-Image"

    Important: Segmentation and processing is done Slice-by-Slice.

    """

    def __init__(self, napari_viewer):
        """Initialize widget

        Parameters
        ----------
        napari_viewer : napari.utils._proxies.PublicOnlyProxy
            public proxy for the napari viewer object
        """
        super().__init__()

        self.logger = setup_log("CziMetaData")

        self.viewer = napari_viewer

        # set default values
        self.min_overlap_ui: int = 128
        self.model_metadata = None
        self.czann_file: str = "mymodel.czann"
        self.use_gpu: bool = True
        self.batch_size: int = 4
        self.random_object_colors: bool = True
        self.convert_rgb_to_bgr: bool = False
        self.tiling_method = TileMethod.CZTILE
        self.merge_method = SupportedWindow.none

        # create a layout
        self.setLayout(QVBoxLayout())

        # add a label
        self.layout().addWidget(QLabel("Model File Selection"))

        # define filter based on file extension
        model_extension = (
            "ZEISS model files (*.czann *.czseg);;"
            "CZANN files (*.czann);;"
            "CZSEG files (*.czseg);;"
            "All files (*.*)"
        )

        # create the FileEdit widget and add to the layout and connect it
        self.filename_edit = FileEdit(mode=FileDialogMode.EXISTING_FILE, value="", filter=model_extension)

        self.layout().addWidget(self.filename_edit.native)
        self.filename_edit.line_edit.changed.connect(self._file_changed)

        # add table for model metadata
        self.model_metadata_label = QLabel("Model Metadata")
        self.model_metadata_label.setFont(QFont("Arial", 11, QFont.Normal))

        # setting up background color and border
        # self.model_metadata_label.setStyleSheet("background-color: yellow;border: 1px solid black;")
        self.layout().addWidget(self.model_metadata_label)

        self.model_metadata_table = TableWidget()
        # self.model_metadata_table.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding)
        # self.model_metadata_table.setFixedSize(QSize(275, 250))
        self.layout().addWidget(self.model_metadata_table)

        # add button for reading the model metadata
        self.read_modeldata_btn = QPushButton("Reload Minimum Overlap")
        self.read_modeldata_btn.clicked.connect(self._read_model_metadata)
        self.layout().addWidget(self.read_modeldata_btn)

        # add label and slider for adjusting the minimum overlap
        self.min_overlap_label = QLabel("Adjust Minimum Overlap")
        self.min_overlap_slider = Slider(
            orientation="horizontal",
            label="Minimum Overlap",
            value=128,
            min=1,
            max=256,
            step=1,
            readout=True,
            tooltip="Adjust the desired min. TileOverlap",
            tracking=False,
        )

        self.layout().addWidget(self.min_overlap_label)
        self.layout().addWidget(self.min_overlap_slider.native)
        self.min_overlap_slider.changed.connect(self._update_min_overlap)
        self.min_overlap_slider.enabled = False

        # add tiling options
        self.tiling_method_label = QLabel("Tiling Method")
        self.tiling_method_edit = ComboBox(value=TileMethod.CZTILE, choices=TileMethod)
        self.tiling_method_edit.enabled = False
        self.layout().addWidget(self.tiling_method_label)
        self.layout().addWidget(self.tiling_method_edit.native)
        self.tiling_method_edit.changed.connect(self._tiling_method_changed)

        # add merge window options
        self.merge_method_label = QLabel("Merge Method (Tiler)")
        self.merge_method_edit = ComboBox(value=SupportedWindow.none, choices=SupportedWindow)
        self.merge_method_edit.enabled = False
        self.layout().addWidget(self.merge_method_label)
        self.layout().addWidget(self.merge_method_edit.native)
        self.merge_method_edit.changed.connect(self._merge_method_changed)

        # add the checkbox the use the GPU for the inference
        self.use_gpu_checkbox = CheckBox(
            name="Use GPU (experimental)",
            visible=True,
            enabled=True,
            value=self.use_gpu,
        )
        self.use_gpu_checkbox.clicked.connect(self._use_gpu_changed)
        self.layout().addWidget(self.use_gpu_checkbox.native)

        # add batch size slider for GPU inference optimization
        self.batch_size_label = QLabel("Batch Size (GPU Optimization)")
        self.batch_size_slider = Slider(
            orientation="horizontal",
            label="Batch Size",
            value=4,
            min=1,
            max=256,
            step=1,
            readout=True,
            tooltip="Tiles processed per GPU forward pass. Larger tiles or 4 GB GPUs often need 1-4; increase while GPU memory allows.",
            tracking=False,
        )

        self.layout().addWidget(self.batch_size_label)
        self.layout().addWidget(self.batch_size_slider.native)
        self.batch_size_slider.changed.connect(self._batch_size_changed)

        self.random_object_colors_checkbox = CheckBox(
            name="Random object colors",
            visible=True,
            enabled=True,
            value=self.random_object_colors,
        )
        self.random_object_colors_checkbox.clicked.connect(self._random_object_colors_changed)
        self.layout().addWidget(self.random_object_colors_checkbox.native)

        # Checkbox: convert RGB→BGR before inference.
        # Auto-set from PixelType metadata when a .czseg model is loaded.
        # Must be enabled manually for .czann/.czmodel models trained on BGR input (ZEN SplitRgb).
        self.convert_rgb_to_bgr_checkbox = CheckBox(
            name="Convert RGB\u2192BGR (for ZEN Bgr24 models)",
            visible=True,
            enabled=True,
            value=self.convert_rgb_to_bgr,
            tooltip=(
                "Enable when the model was trained on BGR-ordered channels (ZEN SplitRgb / Bgr24).\n"
                "czitools always delivers color CZI images in RGB order; this flag reverses channels\n"
                "before inference so they match the training distribution.\n"
                "Auto-set from PixelType metadata for .czseg models."
            ),
        )
        self.convert_rgb_to_bgr_checkbox.clicked.connect(self._convert_rgb_to_bgr_changed)
        self.layout().addWidget(self.convert_rgb_to_bgr_checkbox.native)

        # check GPU availability and disable checkbox if not usable
        self._gpu_available = is_gpu_available()
        if self._gpu_available:
            self.logger.info(
                "CUDAExecutionProvider is listed by ONNX Runtime. "
                "A model-specific CUDA preflight will run before inference; "
                "CPU will be used automatically if CUDA cannot execute the model."
            )
        else:
            self.use_gpu = False
            self.use_gpu_checkbox.value = False
            self.use_gpu_checkbox.enabled = False
            self.logger.warning(
                "GPU support is not available in the current environment. "
                "Using CPU for inference. "
                "To enable GPU, install the GPU extra: pip install napari-czann-segment[gpu] "
                "and ensure CUDA toolkit + cuDNN are installed (e.g. via conda)."
            )

        # make a combo box for selecting the image layers
        self.layer_combos = []
        self.image_layer_combo = self.add_image_combo_box("Image Layer")

        # if the user adds or removes layers - update the combo box
        self.viewer.layers.events.inserted.connect(self._reset_layer_options)
        self.viewer.layers.events.removed.connect(self._reset_layer_options)

        # make button for reading the model metadata
        self.segment_btn = PushButton(
            name="Segment or Process selected Image Layer",
            visible=True,
            enabled=False,
        )
        self.segment_btn.clicked.connect(self._segment)
        self.layout().addWidget(self.segment_btn.native)

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
        new_layer_combo.addItems([layer.name for layer in self.viewer.layers if isinstance(layer, Image)])
        combo_row.layout().addWidget(new_layer_combo)

        # saving to a list, so we can iterate through all combo boxes to reset choices
        self.layer_combos.append(new_layer_combo)
        self.layout().addWidget(combo_row)

        # returning the combo box, so we know which is which when we click run
        return new_layer_combo

    def _read_model_metadata(self):
        """Get model metadata and store them

        This method extracts the model information and path from the czann_file,
        unpacks the model using the appropriate parser (CZANN or CZSEG), and stores
        the model metadata and dictionary representation of the metadata.

        It also updates the model metadata table, sets the min_overlap_ui and min_overlap_slider values,
        enables/disables certain buttons and sliders based on the tiling_method,
        and displays a warning if the model type is regression with output shape
        (Y, X, 1).
        """

        # extract the model information and path
        with tempfile.TemporaryDirectory() as temp_path:

            # Determine file type and use appropriate parser
            file_extension = Path(self.czann_file).suffix.lower()

            if file_extension == ".czseg":
                self.logger.info("Detected CZSEG file format")
                self.model_metadata, self.model_path, expects_bgr = extract_czseg_model(
                    path=self.czann_file, target_dir=Path(temp_path)
                )
                # Auto-set BGR flag from PixelType metadata; keep checkbox editable for override.
                self.convert_rgb_to_bgr = expects_bgr
                self.convert_rgb_to_bgr_checkbox.value = expects_bgr
                self.logger.info(f"Auto-detected convert_rgb_to_bgr={expects_bgr} from .czseg PixelType metadata.")
            elif file_extension in [".czann", ".czmodel"]:
                self.logger.info(f"Detected {file_extension.upper()} file format")
                self.model_metadata, self.model_path = extract_czann_model(
                    path=self.czann_file, target_dir=Path(temp_path)
                )
                # No PixelType metadata in .czann/.czmodel — leave the checkbox at its current value.
                self.logger.info(
                    "convert_rgb_to_bgr not auto-detectable for .czann/.czmodel. "
                    f"Current setting: {self.convert_rgb_to_bgr}"
                )
            else:
                self.logger.error(f"Unsupported file format: {file_extension}")
                warnings.warn(f"Unsupported model file format: {file_extension}")
                return

        # get model metadata as dictionary
        self.model_metadata_dict = self.model_metadata._asdict()
        self.model_metadata_table.update_model_metadata(self.model_metadata_dict)
        self.model_metadata_table.update_style()

        # get the specification from the model metadata
        self.min_overlap_ui = self.model_metadata.min_overlap[0]
        self.min_overlap_slider.value = self.min_overlap_ui

        # enable the button and check
        self.segment_btn.enabled = True
        self.min_overlap_slider.enabled = True
        self.tiling_method_edit.enabled = True
        self._tiling_method_changed()

        if self.tiling_method is TileMethod.CZTILE:
            self.merge_method_edit.enabled = False
            self.merge_method = SupportedWindow.none

        if self.tiling_method is TileMethod.TILER:
            self.merge_method_edit.enabled = True

        if self.tiling_method is TileMethod.RYOMEN:
            self.merge_method_edit.enabled = False
            self.merge_method = SupportedWindow.none

        if self.model_metadata.model_type == ModelType.REGRESSION and self.model_metadata.output_shape[-1] > 1:

            warnings.warn("Only Regression Models with output shape (Y, X, 1) are currently supported.")
            self.segment_btn.enabled = False

    def _segment(self):
        """
        Run the segmentation or processing.

        This method performs segmentation or processing on the selected image layer in the napari viewer.
        It retrieves the necessary parameters from the UI and uses them to call the appropriate prediction function.
        After the prediction is done, it adds the resulting segmentation or processed image as a new layer to the viewer.

        Returns:
            None
        """
        self.logger.info("Run the segmentation or processing.")

        # deactivate the button
        self.segment_btn.enabled = False

        # grab the layer using the combo box item text as the layer name
        img_layer = self.viewer.layers[self.image_layer_combo.currentText()]

        self.logger.info("CZANN Modelfile: " + self.czann_file)
        self.logger.info("CZANN ModelType: " + str(self.model_metadata.model_type))
        self.logger.info("Minimum Tile Overlap: " + str(self.min_overlap_ui))
        self.logger.info("Use GPU acceleration: " + str(self.use_gpu))
        self.logger.info("Batch Size: " + str(self.batch_size))
        self.logger.info("Random object colors: " + str(self.random_object_colors))

        self.logger.info(f"Convert RGB\u2192BGR: {self.convert_rgb_to_bgr}")

        if self.model_metadata.model_type == ModelType.SINGLE_CLASS_SEMANTIC_SEGMENTATION:

            modeldata, seg_complete = predict_ndarray(
                self.czann_file,
                img_layer.data,
                border=self.min_overlap_ui,
                use_gpu=self.use_gpu,
                do_rescale=_resolve_do_rescale(getattr(self.model_metadata, "scaling", None)),
                tiling_method=self.tiling_method,
                merge_window=self.merge_method,
                batch_size=self.batch_size,
                convert_rgb_to_bgr=self.convert_rgb_to_bgr,
            )

            self.logger.info(f"Input Data Shape: {img_layer.data.shape}")
            self.logger.info(f"Output Data Shape: {seg_complete.shape}")

            # create a list of label values
            label_values = list(range(1, len(modeldata.classes) + 1))

            # get individual outputs for all classes from the label image
            for c in range(len(modeldata.classes)):

                # get the pixels for which the value is equal to current class value
                self.logger.info("Class Name: " + modeldata.classes[c] + " Prediction Pixel Value: " + str(c))

                if self.random_object_colors:
                    labels_current_class = label_nd(seg_complete, labelvalue=label_values[c])
                else:
                    labels_current_class = (seg_complete == label_values[c]).astype(np.uint8)

                # add new image layer
                self.viewer.add_labels(
                    labels_current_class,
                    name=f"{img_layer.name}_" + modeldata.classes[c],
                    # num_colors=256,
                    scale=img_layer.scale,
                    opacity=0.7,
                    blending="translucent",
                )

        if self.model_metadata.model_type == ModelType.REGRESSION:

            modeldata, processed_image = predict_ndarray(
                self.czann_file,
                img_layer.data,
                border=self.min_overlap_ui,
                use_gpu=self.use_gpu,
                do_rescale=False,
                merge_window=self.merge_method,
                batch_size=self.batch_size,
            )

            self.logger.info(f"Input Data Shape: {img_layer.data.shape}")
            self.logger.info(f"Output Data Shape: {processed_image.shape}")

            # add new image layer
            # add channel to napari viewer
            self.viewer.add_image(
                processed_image,
                name=f"{img_layer.name}_processed",
                scale=img_layer.scale,
            )

        # reactivate the button
        self.segment_btn.enabled = True

    def _update_min_overlap(self):
        """
        Update the minimum overlap value based on the slider value.

        This method is called when the slider for minimum overlap is adjusted. It updates the `min_overlap_ui` attribute
        with the current value of the slider and then checks if the minimum overlap is valid using the `_check_min_overlap`
        method.
        """
        self.min_overlap_ui = self.min_overlap_slider.value
        self._check_min_overlap()

    def _file_changed(self):
        """
        Callback method triggered when the file selection is changed.
        It updates the `czann_file` attribute with the selected file path,
        replaces backslashes with forward slashes, and logs the model path.
        It also updates the model metadata by calling the `_read_model_metadata` method.
        """
        self.czann_file = str(self.filename_edit.value.absolute()).replace("\\", "/").replace("//", "/")
        self.logger.info("Model Path: " + self.czann_file)

        # update the model metadata
        self._read_model_metadata()

    def _reset_layer_options(self, event):
        """Clear existing combo boxes and repopulate
        Parameters
        ----------
        event : event
            Clear existing combo box items and query viewer for all image layers
        """
        for combo in self.layer_combos:
            combo.clear()
            combo.addItems([layer.name for layer in self.viewer.layers if isinstance(layer, Image)])

    def _check_min_overlap(self):
        """
        Check the minimum overlap value and adjust if necessary.

        The minimum border width must be less than half the tile size. If the current
        minimum overlap value is greater than or equal to half the tile size, it will
        be adjusted to be half the tile size minus 1.

        Returns:
            None
        """
        # Minimum border width must be less than half the tile size.
        min_tilesize = min(
            self.model_metadata.input_shape[0],
            self.model_metadata.input_shape[1],
        )

        # check
        if self.min_overlap_ui >= np.round(min_tilesize / 2, 0):
            self.min_overlap_ui = int(np.round(min_tilesize / 2, 0) - 1)
            self.logger.info("Minimum border width must be less than half the tile size.")
            self.logger.info("Adjusted minimum overlap : " + self.min_overlap_ui)
            self.min_overlap_slider.value = self.min_overlap_ui

    def _use_gpu_changed(self):
        """
        Callback method triggered when the 'use_gpu_checkbox' value changes.
        Updates the 'use_gpu' attribute based on the checkbox value and logs the inference device being used.
        """
        if self.use_gpu_checkbox.value:
            self.use_gpu = True
            self.logger.info("Use GPU for inference.")
        if not self.use_gpu_checkbox.value:
            self.use_gpu = False
            self.logger.info("Use CPU for inference.")

    def _batch_size_changed(self):
        """
        Callback method triggered when the batch size slider value changes.
        Updates the batch_size attribute and logs the new batch size.
        Higher values improve GPU utilization but require more VRAM.
        """
        self.batch_size = self.batch_size_slider.value
        self.logger.info(f"Batch Size: {self.batch_size}")

    def _random_object_colors_changed(self):
        """Update whether connected objects are labelled independently."""
        self.random_object_colors = self.random_object_colors_checkbox.value
        self.logger.info(f"Random object colors: {self.random_object_colors}")

    def _convert_rgb_to_bgr_changed(self):
        """Update RGB→BGR conversion flag from the checkbox."""
        self.convert_rgb_to_bgr = self.convert_rgb_to_bgr_checkbox.value
        self.logger.info(f"Convert RGB\u2192BGR: {self.convert_rgb_to_bgr}")

    def _tiling_method_changed(self):
        """
        Callback method that is called when the tiling method is changed.

        Updates the tiling method attribute and performs necessary actions based on the selected tiling method.
        """

        self.tiling_method = self.tiling_method_edit.value
        self.logger.info(f"Tiling Method: {self.tiling_method }")

        # if Tiler is used
        if self.tiling_method == TileMethod.TILER:
            self.merge_method_edit.enabled = True
            self.merge_method_edit.value = SupportedWindow.none

        if self.tiling_method == TileMethod.CZTILE:
            self.merge_method_edit.enabled = False
            self.merge_method_edit.value = SupportedWindow.none

        if self.tiling_method == TileMethod.RYOMEN:
            self.merge_method_edit.enabled = False
            self.merge_method_edit.value = SupportedWindow.none

    def _merge_method_changed(self):
        """
        Callback method called when the merge method is changed.

        Updates the `merge_method` attribute and logs the new merge method.

        Parameters:
        None

        Returns:
        None
        """
        self.merge_method = self.merge_method_edit.value
        self.logger.info(f"Merge Method (Tiler): {self.merge_method }")


if __name__ == "__main__":

    viewer = napari.Viewer()
    napari.run()
