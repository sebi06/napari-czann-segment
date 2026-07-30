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
- Toggle **Convert RGB→BGR** if the model was trained on BGR-ordered channels — see [RGB vs. BGR channel order](#rgb-vs-bgr-channel-order) below.
- Press **Segment Selected Image Layer** to run the segmentation.

![Napari - Image successfully segmented](https://github.com/sebi06/napari-czann-segment/raw/main/readme_images/napari_czann3.png)

A successful is obviously only the starting point for further image analysis steps to extract the desired numbers from the segmented image.
Another example is shown below demonstrating a simple "Grain Size Analysis" using a deep-learning model trained on [APEER] used in [napari]

![Napari - Simple Grain Size Analysis](https://github.com/sebi06/napari-czann-segment/raw/main/readme_images/grainsize_czann_napari.png)

### Remarks

> **IMPORTANT**: Currently the plugin only supports using models trained on a **single channel** image (grayscale) or a **3-channel color** image (RGB / BGR). Multi-channel fluorescence images with more than 3 channels are not supported. Make sure that during training on [APEER] or elsewhere the correct input images are used.
> It is quite simple to accidentally train a single RGB image (3 channels) and notice only later that the model will not work as expected inside [napari].

### RGB vs. BGR channel order

[napari] always loads and stores color images in **RGB** order (red-green-blue). However, models trained using ZEN's *SplitRgb* pipeline or saved with the `Bgr24` / `Bgra32` pixel type expect channels in **BGR** order (blue-green-red). Running such a model without channel reordering will produce incorrect or degraded segmentation results.

The plugin handles this with the **Convert RGB→BGR** checkbox:

| Model / file type                               | Behavior                                                                                     |
| :---------------------------------------------- | :------------------------------------------------------------------------------------------- |
| `.czseg` with `Bgr24` or `Bgra32` pixel type    | Auto-detected — checkbox is set automatically                                                |
| `.czseg` with `Rgb24` / `Gray8` / `Gray16` etc. | Auto-detected — no conversion needed                                                         |
| `.czann` / `.czmodel`                           | Auto-detection not possible — toggle checkbox manually if the model was trained on BGR input |

**When do you need it?**

- The model was trained inside ZEN (ZEN SplitRgb path) → enable the checkbox.
- The model was trained on standard RGB images (e.g. APEER, own Jupyter notebook) → leave the checkbox off.
- The input image is grayscale (single channel) → the checkbox has no effect.

The checkbox can always be toggled manually to override the auto-detected value.

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

GPU acceleration uses [ONNX-GPU] and requires an **NVIDIA GPU** with the correct CUDA runtime libraries. For the ONNX Runtime versions targeted by this project, the `[gpu]` extra installs pinned pip `nvidia-*-cu12` runtime packages, using a CUDA 12.4 + cuDNN 9.1 stack that still works on older NVIDIA GPUs such as GTX 10xx / Pascal.

**Why the conda environment file is recommended:**

The conda environment file ([napari-env.yml](napari-env.yml)) already has the **correct versions of CUDA and cuDNN** tested and working with `onnxruntime-gpu`. You don't need to guess which versions to install:

    conda env create --file napari-env.yml

Then verify GPU support:

    python -c "import onnxruntime; print(onnxruntime.get_available_providers())"

You should see `CUDAExecutionProvider` in the list.

**If you must install manually:**

Different versions of `onnxruntime-gpu` require different CUDA/cuDNN versions. Check the [ONNX Runtime CUDA Provider documentation](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements) for your version's requirements.

For example, for `onnxruntime-gpu >= 1.21`, you need CUDA 12.x and cuDNN 9.x. The `[gpu]` extra pins tested runtime wheels:

1. **If you previously installed with `[cpu]`, uninstall onnxruntime first**:

       pip uninstall onnxruntime -y

2. **Install the GPU extra**:

       pip install napari-czann-segment[gpu]

3. **Verify GPU support**:

       python -c "import onnxruntime; print(onnxruntime.get_available_providers())"

   You should see `CUDAExecutionProvider` in the list. If you only see `CPUExecutionProvider`, the CUDA libraries are missing or incompatible — check the [requirements](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements) and reinstall.

4. **(Alternative) Use conda packages** if you prefer not to use pip-provided CUDA runtime wheels:

       conda install nvidia::cuda-runtime nvidia::cudnn

5. **Use the repo environment file** for a tested setup. See the example [conda environment YAML](napari-env.yml):

       conda env create --file napari-env.yml

**Note:** If you have PyTorch with CUDA installed, onnxruntime-gpu (>= 1.21) can automatically reuse PyTorch's CUDA/cuDNN DLLs via its `preload_dlls()` mechanism. The plugin calls this automatically at startup.

### CUDA preflight check

Before running GPU inference, the plugin performs a one-time **CUDA preflight** per model: it creates a CUDA ONNX session and runs a single dummy inference in a **separate subprocess**. This is intentional — some CUDA/cuDNN/cuBLAS mismatches abort the process from native code (an error Python cannot catch), so running the probe out-of-process lets napari fall back to CPU safely instead of crashing. The result is cached per model for the session.

The **first** CUDA session on a machine can be slow to initialize (cuDNN/cuBLAS DLL loading, kernel JIT compilation, driver warm-up). If the preflight does not finish within its timeout, the plugin logs a warning and falls back to CPU for that session. If your GPU is healthy but simply slow to warm up, increase the timeout via an environment variable **before launching napari**:

    # Windows (PowerShell) - timeout in seconds
    $env:NAPARI_CZANN_CUDA_PREFLIGHT_TIMEOUT = "300"

    # Linux / macOS
    export NAPARI_CZANN_CUDA_PREFLIGHT_TIMEOUT=300

The default is **180 seconds**.

### Troubleshooting GPU

- **Only seeing `CPUExecutionProvider` after installing `[gpu]`**: You likely have both `onnxruntime` and `onnxruntime-gpu` installed (they conflict). **Solution**: Run `pip uninstall onnxruntime -y` to remove the CPU version, keeping only the GPU version.
- **Switching from `[cpu]` to `[gpu]`**: First uninstall the CPU version: `pip uninstall onnxruntime -y`, then install with `pip install napari-czann-segment[gpu]`.
- **`libcudnn.so.9`, `cublasLt64_12.dll`, or `cufft64_11.dll` not found**: CUDA runtime libraries are missing. The easiest fix is ensuring you installed with `[gpu]` extra, which includes them. Alternatively, install via conda: `conda install nvidia::cuda-runtime nvidia::cudnn`.
- **`CUDNN_FE failure 11` with `no kernel image is available for execution on the device`**: The installed pip `nvidia-*-cu12` wheels are probably too new for your GPU generation. Reinstall the local package with the pinned `[gpu]` extra:

       pip install --force-reinstall -e .[gpu]

- **`Failed to allocate memory for requested buffer` during a Conv node**: The GPU ran out of memory during inference. Lower the batch size in the plugin UI; the plugin also retries GPU inference with smaller batches and falls back to CPU if one tile still cannot fit.
- **Plugin shows "GPU support is not available"**: Check the napari log output for detailed diagnostics. The plugin always falls back to CPU safely.
- **`CUDA preflight ... timed out` warning / GPU falls back to CPU**: The one-time CUDA warm-up exceeded the preflight timeout (default 180 s). If your GPU is healthy, raise it by setting `NAPARI_CZANN_CUDA_PREFLIGHT_TIMEOUT` (seconds) before launching napari — see [CUDA preflight check](#cuda-preflight-check).
- **CUDA version mismatch**: `onnxruntime-gpu` requires specific CUDA versions. Check the [ONNX Runtime GPU requirements](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements).

## For developers

- **Please clone this repository first using your favorite tool.**

- **Ideally one creates a new [conda] environment or use an existing environment that already contains [Napari].**

Feel free to create a new environment using the example [YAML](napari-env.yml) file at your own risk:

    cd the-github-repo-with-YAML-file
    conda env create --file napari-env.yml
    conda activate napari-env

- **Install the plugin locally**

Please run the following command:

    pip install -e .

To install latest development version:

    pip install git+https://github.com/sebi06/napari_czann_segment.git

### Running tests

Run the automated test suite with:

    pytest

The `_tests/` directory also contains **manual diagnostic scripts** (`test_color_conversion.py`, `diagnostic_color_test.py`, `debug_tiling.py`) that perform full end-to-end inference on real CZI files and take several minutes. These are automatically **skipped** by `pytest`. To run them manually:

    python src/napari_czann_segment/_tests/test_color_conversion.py
    python src/napari_czann_segment/_tests/diagnostic_color_test.py
    python src/napari_czann_segment/_tests/debug_tiling.py

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
