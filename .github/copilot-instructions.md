# Copilot Instructions — napari-czann-segment

## Project overview

`napari-czann-segment` is a [napari](https://napari.org) plugin for semantic segmentation and image processing using ONNX models packaged as `*.czann` files (ZEISS model format). It supports CPU and optional GPU inference via `onnxruntime`.

## Repository layout

```
src/napari_czann_segment/   # main package
    dock_widget.py           # napari widget (Qt / magicgui)
    onnx_inference.py        # ONNX session management, GPU detection, inference
    predict.py               # tiled prediction orchestration (cztile / Tiler / ryomen)
    process_nd.py            # post-processing (label_nd)
    utils.py                 # helpers, enums (TileMethod, SupportedWindow)
    get_testdata.py          # test-data locator (models + images in _data/)
    _sample_data.py          # napari sample-data hook
    napari.yaml              # napari manifest
    _data/                   # bundled .czann models and test images
    _tests/                  # pytest test suite
setup.cfg                    # package metadata & dependencies
pyproject.toml               # build system (setuptools + setuptools_scm)
tox.ini                      # tox / GitHub Actions matrix
```

## Python & dependencies

- **Python ≥ 3.11** (CI matrix: 3.11, 3.12, 3.13).
- Base dependencies: `napari ≥ 0.7`, `cztile ≥ 2`, `czmodel ≥ 5`, `onnxruntime`, `tiler`, `ryomen`.
- GPU extra: `pip install napari-czann-segment[gpu]` → installs `onnxruntime-gpu[cuda,cudnn]`.
- Testing extra: `pip install napari-czann-segment[testing]` → `pytest`, `pytest-cov`, `pytest-qt`, `bioio`, etc.
- `numpy`, `magicgui`, `qtpy` are **transitive** via napari — do not add them to `install_requires`.

## Code style

- Formatter: **Black**, line length **79**.
- Import sorting: **isort** (profile `black`, line length 79).
- Type hints are used throughout; follow existing conventions.
- Use `logging` (module-level `logger`) — not `print()` — for runtime diagnostics.

## ONNX / GPU conventions

- `onnxruntime` import is guarded with try/except; the module-level flag `ONNXRUNTIME_AVAILABLE` tracks availability.
- `ManagedOnnxSession` is the **only** way to create inference sessions. It automatically falls back from CUDA to CPU on failure.
- `is_gpu_available()` is the canonical GPU-readiness check.
- CUDAExecutionProvider options must include `cudnn_conv_algo_search: "EXHAUSTIVE"` and `cudnn_conv_use_max_workspace: "1"`.
- On Windows, conda CUDA DLL directories are registered via `os.add_dll_directory()` at module import time.

## Testing

- Test runner: **pytest** (via `tox` in CI).
- Tests live in `src/napari_czann_segment/_tests/`.
- All tests **must** work in CI (no GPU, no display). Use `@pytest.mark.skipif` with `CI` / `GITHUB_ACTIONS` env vars for GPU-only tests.
- Test models and images are bundled in `_data/`; access them through `get_testdata.get_modelfile()` / `get_testdata.get_imagefile()`.
- Run locally: `pytest` from the repo root.

## Common pitfalls

- **Do not** install both `onnxruntime` and `onnxruntime-gpu` — they conflict. The base package provides CPU; the `[gpu]` extra replaces it.
- macOS has no NVIDIA GPU support; the `[gpu]` extra will fail to install there.
- `czmodel.core.util._extract_model.extract_czann_model` is the correct API for unpacking `.czann` files (not the deprecated `DefaultConverter`).
