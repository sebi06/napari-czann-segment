# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-07-16

### Added

- **CZSEG model support**: new `czseg_parser.py` can load ZEISS ZEN `*.czseg`
  model files, parse their XML metadata (classes, per-class RGB colors,
  tile/border sizes), and infer tile size from the embedded ONNX model when the
  XML omits it. The model file picker now accepts `*.czann` and `*.czseg`.
- Batched ONNX inference via `OnnxInferencer(batch_size=...)` for faster
  prediction on multi-tile images.
- Out-of-process CUDA preflight probe that verifies a usable CUDA session before
  running inference, with automatic and safe fallback to CPU.
- New tests covering CZSEG parsing, RGB/BGR color conversion, `predict_ndarray`
  axis handling, and batching.

### Changed

- Major rewrite of `predict.py` improving tiling, axis handling, and support for
  `xarray.DataArray` inputs.
- `onnxruntime` is now imported defensively in `czseg_parser.py` (matching
  `onnx_inference.py`) so a failed native DLL load on Windows CI no longer
  crashes module import or test collection.
- CI: `onnxruntime` is installed through the `testing` extra for all platforms
  (Linux, macOS, Windows); redundant per-OS install steps were removed.
- Replaced the `env_napari_czann_segment.yml` environment file with
  `napari-env.yml`.

### Fixed

- BGR/RGB channel-order handling for models trained on BGR data, preventing
  incorrect segmentation results.
- Tiling seam/edge artifacts in stitched prediction output.
- GPU usage and CUDA preflight robustness on machines with mismatched
  CUDA/cuDNN libraries.

## [0.0.23] - 2026-05-17

### Fixed

- `TableWidget.__init__` now calls `super().__init__()` instead of
  `super(QWidget, self).__init__()`, resolving a `TypeError` raised by PySide6
  (*"PySide6.QtCore.QObject isn't a direct base class of TableWidget"*) when
  opening the plugin dock widget.

### Changed

- Added `Operating System :: MacOS` to the PyPI classifiers in `setup.cfg`.

## [0.0.22] - 2026-03-29

### Fixed

- CI: align PyPI publish workflow with TestPyPI workflow.
- Skip `onnxruntime` tests when import fails on Windows CI.
- Suppress `onnxruntime` access violation on Windows CI.

### Added

- Robust GPU/CPU inference with automatic CUDA → CPU fallback, expanded test
  coverage, and dependency cleanup.
- Upgrade GitHub Actions runners to Node.js 24 compatible action versions.

## [0.0.21] - 2025-10-26

### Fixed

- Multiple CI pipeline fixes.
- Compatibility updates for `cztile` 2.0.
- Remove PyTorch dependency.

## [0.0.20] - 2025-09-27

### Changed

- `predict_ndarray` now also accepts `xarray.DataArray` as input.

## [0.0.19] - 2025-09-26

### Fixed

- CI fixes; limit supported Python versions in test matrix.

### Changed

- Replaced `aicsimageio` with `bioio` for image I/O.
- Added support for Python 3.11 and 3.12.

## [0.0.18] - 2023-09-22

### Changed

- General updates and compatibility improvements.
- Added Python 3.11 / 3.12 support in CI matrix.

## [0.0.17] - 2023-05-18

### Changed

- Only `czmodel[pytorch]` is now supported (dropped other model backends).
- Various code and CI updates.

## [0.0.16] - 2022-10-04

### Changed

- Updated `setup.cfg` for `czmodel` 5.0.0 release.
- Internal adaptations for the `czmodel` 5.0.0 API.

## [0.0.15] - 2022-07-22

### Changed

- Set development status classifier to *Alpha*.

## [0.0.14] - 2022-07-22

### Changed

- README updates.

## [0.0.13] - 2022-07-18

### Added

- Table widget to display CZANN model metadata in the dock widget.

### Changed

- README and documentation updates.

## [0.0.12] - 2022-07-13

### Changed

- General updates and fixes following community PR contributions.

## [0.0.11] - 2022-07-11

### Changed

- Use a smaller GIF in the README for faster page loading.

## [0.0.10] - 2022-07-11

### Changed

- General updates.

## [0.0.9] - 2022-07-11

### Added

- Initial TestPyPI release.
