# -*- coding: utf-8 -*-

#################################################################
# File        : ndarray_prediction.py
# Author      : sebi06, Team Enchilada
#
# Disclaimer: This code is purely experimental. Feel free to
# use it at your own risk.
#
#################################################################

from napari_czann_segment import get_testdata
from napari_czann_segment import predict, process_nd
from aicsimageio import AICSImage
import napari

# get data to test the functionality
name_czann = "PGC_20X_nucleus_detector.czann"
img_name = "PGC_20X.ome.tiff"

# get the correct file path for the sample data
czann_file = get_testdata.get_modelfile(name_czann)
img_path = get_testdata.get_imagefile(img_name)

# use the GPU for inference - requires onnxruntime-gpu !!!
use_gpu = False

# read the CZI using AICSImageIO and aicspylibczi
aics_img = AICSImage(img_path)
print("Dimension Original Image:", aics_img.dims)
print("Array Shape Original Image:", aics_img.shape)

if aics_img.physical_pixel_sizes.X is None:
    aics_img.physical_pixel_sizes.X = 1.0
if aics_img.physical_pixel_sizes.Y is None:
    aics_img.physical_pixel_sizes.Y = 1.0

# get the scaling - will be applied to the segmentation outputs
scale = (aics_img.physical_pixel_sizes.X, aics_img.physical_pixel_sizes.Y)

# read the image data as numpy or dask array
img = aics_img.get_image_data()
# img = aics_img.get_image_dask_data()

# create list for the layers
napari_layers = []

# add the image as a layer to the napari viewer
viewer = napari.Viewer()
img_layer = viewer.add_image(img,
                             name="original",
                             scale=scale,
                             blending="translucent")

modeldata, seg_complete = predict.predict_ndarray(czann_file,
                                                  img,
                                                  border="auto",
                                                  use_gpu=use_gpu)

# create a list of label values
label_values = list(range(1, len(modeldata.classes) + 1))

# get individual outputs for all classes from the label image
for c in range(len(modeldata.classes)):
    # get the pixels for which the value is equal to current class value
    print("Class Name:", modeldata.classes[c], "Prediction Pixel Value:", c)

    # get all pixels with a specific value as boolean array, convert to numpy array and label
    labels_current_class = process_nd.label_nd(seg_complete,
                                               labelvalue=label_values[c])

    # add new image layer
    seg_layer = viewer.add_labels(labels_current_class,
                                  name=f"{img_layer.name}_" + modeldata.classes[c],
                                  num_colors=256,
                                  scale=scale,
                                  opacity=0.7,
                                  blending="translucent")

    napari_layers.append(seg_layer)

# add the original image as a layer to the Napari viewer
napari_layers.append(img_layer)

print("Done.")

napari.run()
