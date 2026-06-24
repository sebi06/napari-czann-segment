# napari-czann-segment

[![License](https://img.shields.io/pypi/l/napari-czann-segment.svg?color=green)](https://github.com/sebi06/napari-czann-segment/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-czann-segment.svg?color=green)](https://pypi.org/project/napari-czann-segment)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-czann-segment.svg?color=green)](https://python.org)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-czann-segment)](https://napari-hub.org/plugins/napari-czann-segment)

Semantic Segmentation of multidimensional images using Deep Learning ONNX models packaged as *.czann files.

----------------------------------

This [napari] plugin was generated with [Cookiecutter] using [@napari]'s [cookiecutter-napari-plugin] template.

![Train on APEER and use model in Napari](https://github.com/sebi06/napari-czann-segment/raw/main/readme_images/Train_APEER_run_Napari_CZANN_no_highlights_small.gif)

## Installation

Before installing, please setup a conda environment. If you have never worked with conda environments, go through [this tutorial](https://biapol.github.io/blog/johannes_mueller/anaconda_getting_started/) first.

You can then install `napari-czann-segment` via [pip]. **You must choose either CPU or GPU support**:

**For CPU inference** (works on all platforms):

    pip install napari-czann-segment[cpu]

**For GPU inference** (Windows/Linux with NVIDIA GPU only):

    pip install napari-czann-segment[gpu]

See [CPU vs. GPU Inference](#cpu-vs-gpu-inference) for detailed GPU setup instructions.

## What does the plugin do

The plugin allows you to:

- Use a *.czann file containing the Deep Neural Network (ONNX) for semantic segmentation and metadata
- Segmentation will be applied per 2D plane for all dimensions
- Processing larger multidimensional images it uses the [cztile] package to chunk the individual 2d arrays using a specific overlap.
- multidimensional images will be processed plane-by-plane

## What does the plugin NOT do

**Before one can actually use a model it needs to be trained, which is NOT done by this plugin**.

There are two main ways hwo such a model can be created:

- Train the segmentation model fully automated on [APEER] and download the *.czann file
- Train your model in a Jupyter notebook etc. and package it using the [czmodel] python package as an *.czann

## Using this plugin

### Sample Data

A test image and a *.czann model file can be downloaded [here](https://github.com/sebi06/napari-czann-segment/tree/main/src/napari_czann_segment/_data).

- `PGC_20X.ome.tiff` --> use `PGC_20X_nucleus_detector.czann` to segment

In order to use this plugin the user has to do the following things:

- Open the image using "File - Open Files(s)" (requires [napari-bioio] plugin).
- Click **napari-czann-segment: Segment with CZANN model** in the "Plugins" menu.
- **Select a czann file** to use the model for segmentation.
- metadata of the model will be shown (see example below)

| Parameter    | Value                                        | Explanation                                        |
| :----------- | :------------------------------------------- | -------------------------------------------------- |
| model_type   | ModelType.SINGLE_CLASS_SEMANTIC_SEGMENTATION | see: [czmodel] for details                         |
| input_shape  | [1024, 1024, 1]                              | tile dimensions of model input                     |
| output_shape | [1024, 1024, 3]                              | tile dimensions of model output                    |
| model_id     | ba32bc6d-6bc9-4774-8b47-20646c7cb838         | unique GUID for that model                         |
| min_overlap  | [128, 128]                                   | tile overlap used during training (for this model) |
| classes      | ['background', 'grains', 'inclusions']       | available classes                                  |
| model_name   | APEER-trained model                          | name of the model                                  |

![Napari - Image loaded and czann selected](https://github.com/sebi06/napari-czann-segment/raw/main/readme_images/napari_czann1.png)

- Adjust the **minimum overlap** for the tiling (optional, see [cztile] for details).
- Select the **layer** to be segmented.
- Toggle **Use GPU for inference** checkbox to enable / disable using a GPU (Nvidia) for the segmentation (experimental feature).
- Press **Segment Selected Image Layer** to run the segmentation.

![Napari - Image successfully segmented](https://github.com/sebi06/napari-czann-segment/raw/main/readme_images/napari_czann3.png)

A successful is obviously only the starting point for further image analysis steps to extract the desired numbers from the segmented image.
Another example is shown below demonstrating a simple "Grain Size Analysis" using a deep-learning model trained on [APEER] used in [napari]

![Napari - Simple Grain Size Analysis](https://github.com/sebi06/napari-czann-segment/raw/main/readme_images/grainsize_czann_napari.png)

### Remarks

> **IMPORTANT**: Currently the plugin only supports using models trained on a **single channel** image. Therefore, make sure that during the training on [APEER] or somewhere else the correct inputs images are used.
> It is quite simple to train a single RGB image, which actually has three channels, load this image in [napari] and notice only then that the model will not work, because the image will 3 channels inside [napari].

## CPU vs. GPU Inference

### CPU (default - works everywhere)

CPU inference via [ONNX-CPU] works on **all platforms** (Windows, Linux, macOS including Apple Silicon):

    pip install napari-czann-segment[cpu]

When the plugin starts, it checks GPU availability and logs the result. If no GPU is detected, the "Use GPU" checkbox is automatically disabled and all inference runs on CPU.

### macOS

On macOS (both Intel and Apple Silicon), the plugin works with **CPU inference only**. NVIDIA CUDA is not available on macOS, so the `[gpu]` extra should **not** be installed — `onnxruntime-gpu` does not publish macOS wheels and installation will fail. Simply use:

    pip install napari-czann-segment[cpu]

The GPU checkbox will be automatically disabled on macOS.

### GPU (optional - Windows/Linux with NVIDIA GPU)

GPU acceleration uses [ONNX-GPU] and requires an **NVIDIA GPU** with the correct CUDA runtime libraries. These libraries **cannot be installed via pip** — they must come from conda or system CUDA installation.

**Why the conda environment file is recommended:**

The conda environment file ([env_napari_czann_segment.yml](env_napari_czann_segment.yml)) already has the **correct versions of CUDA and cuDNN** tested and working with `onnxruntime-gpu`. You don't need to guess which versions to install:

    conda env create --file env_napari_czann_segment.yml

Then verify GPU support:

    python -c "import onnxruntime; print(onnxruntime.get_available_providers())"

You should see `CUDAExecutionProvider` in the list.

**If you must install manually:**

Different versions of `onnxruntime-gpu` require different CUDA/cuDNN versions. Check the [ONNX Runtime CUDA Provider documentation](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements) for your version's requirements.

For example, for `onnxruntime-gpu >= 1.21`, you need CUDA 12.x and cuDNN 9.x:

1. **Install CUDA runtime libraries via conda** (matching your `onnxruntime-gpu` version):

       conda install nvidia::libcublas=12.4 nvidia::libcufft=12 nvidia::libcudart=12 nvidia::cudnn=9

2. **If you previously installed with `[cpu]`, uninstall onnxruntime first**:

       pip uninstall onnxruntime -y

3. **Install the GPU extra**:

       pip install napari-czann-segment[gpu]

4. **Verify GPU support**:

       python -c "import onnxruntime; print(onnxruntime.get_available_providers())"

   You should see `CUDAExecutionProvider` in the list. If you only see `CPUExecutionProvider`, the CUDA libraries are missing or incompatible — check the [requirements](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements) and reinstall.

4. **(Alternative) Use a conda environment** with system CUDA libraries. See the example [conda environment YAML](env_napari_czann_segment.yml):

       conda env create --file env_napari_czann_segment.yml

**Note:** If you have PyTorch with CUDA installed, onnxruntime-gpu (>= 1.21) can automatically reuse PyTorch's CUDA/cuDNN DLLs via its `preload_dlls()` mechanism. The plugin calls this automatically at startup.

### Troubleshooting GPU

- **Only seeing `CPUExecutionProvider` after installing `[gpu]`**: You likely have both `onnxruntime` and `onnxruntime-gpu` installed (they conflict). **Solution**: Run `pip uninstall onnxruntime -y` to remove the CPU version, keeping only the GPU version.
- **Switching from `[cpu]` to `[gpu]`**: First uninstall the CPU version: `pip uninstall onnxruntime -y`, then install with `pip install napari-czann-segment[gpu]`.
- **`cublasLt64_12.dll` or `cufft64_11.dll` not found**: CUDA runtime libraries are missing. The easiest fix is ensuring you installed with `[gpu]` extra which includes them. Alternatively, install via conda: `conda install nvidia::libcublas=12.4 nvidia::libcufft=11.*`.
- **Plugin shows "GPU support is not available"**: Check the napari log output for detailed diagnostics. The plugin always falls back to CPU safely.
- **CUDA version mismatch**: `onnxruntime-gpu` requires specific CUDA versions. Check the [ONNX Runtime GPU requirements](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements).

## For developers

- **Please clone this repository first using your favorite tool.**

- **Ideally one creates a new [conda] environment or use an existing environment that already contains [Napari].**

Feel free to create a new environment using the example [YAML](env_napari_czann_segment.yml) file at your own risk:

    cd the-github-repo-with-YAML-file
    conda env create --file env_napari_czann_segment.yml
    conda activate napari_czann_segment

- **Install the plugin locally**

Please run the following command:

    pip install -e .

To install latest development version:

    pip install git+https://github.com/sebi06/napari_czann_segment.git

## Contributing

Contributions and Feedback are very welcome.

## License

Distributed under the terms of the [BSD-3] license,
"napari-czann-segment" is free and open source software

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[napari]: https://github.com/napari/napari
[Cookiecutter]: https://github.com/audreyr/cookiecutter
[@napari]: https://github.com/napari
[MIT]: http://opensource.org/licenses/MIT
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[GNU GPL v3.0]: http://www.gnu.org/licenses/gpl-3.0.txt
[GNU LGPL v3.0]: http://www.gnu.org/licenses/lgpl-3.0.txt
[Apache Software License 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[Mozilla Public License 2.0]: https://www.mozilla.org/media/MPL/2.0/index.txt
[cookiecutter-napari-plugin]: https://github.com/napari/cookiecutter-napari-plugin
[file an issue]: https://github.com/sebi06/napari-czann-segment/issues
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/
[czmodel]: https://pypi.org/project/czmodel/
[cztile]: https://pypi.org/project/cztile/
[APEER]: https://www.apeer.com
[napari-bioio-reader]: https://github.com/imcf/napari-bioio-reader
[ONNX-GPU]: https://pypi.org/project/onnxruntime-gpu/
[ONNX-CPU]: https://pypi.org/project/onnxruntime/
[conda]: https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html
[pytorch]: https://pytorch.org/get-started/locally
