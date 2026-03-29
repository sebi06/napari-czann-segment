# PR #10 — Robust GPU/CPU inference, expanded tests, dependency cleanup

**Branch:** `version_0.0.22` → `main`

## Summary

Make ONNX inference robust across CPU-only and GPU environments with automatic
fallback, improve test coverage significantly, and clean up package dependencies.

## Changes

### `onnx_inference.py`
- **CUDA → CPU fallback**: `ManagedOnnxSession.__enter__()` catches provider
  failures and transparently falls back to `CPUExecutionProvider`.
- **`is_gpu_available()`**: canonical check that logs onnxruntime version,
  available providers, and actionable advice when CUDA is missing.
- **Conda DLL registration** (Windows): registers `<CONDA_PREFIX>/bin` and
  `<CONDA_PREFIX>/Library/bin` via `os.add_dll_directory()` so CUDA/cuDNN
  DLLs are found even when napari is launched from a GUI shortcut.
- **`onnxruntime.preload_dlls()`**: called at import time (≥ 1.21) to locate
  CUDA/cuDNN from PyTorch, NVIDIA pip packages, or the system.
- **cuDNN provider options**: `cudnn_conv_algo_search: "EXHAUSTIVE"` and
  `cudnn_conv_use_max_workspace: "1"` to eliminate "Fallback to non-cudnn
  non-fused conv2d" performance warnings.

### `dock_widget.py`
- Startup GPU check via `is_gpu_available()`.
- If no GPU is detected the "Use GPU" checkbox is unchecked and disabled, with
  a log warning including remediation steps.

### `setup.cfg`
- Removed transitive dependencies (`numpy`, `magicgui`, `qtpy`) already
  provided by napari.
- Moved `bioio`, `bioio-ome-tiff`, `bioio-imageio` from base to `[testing]`.
- Added `[gpu]` extra: `onnxruntime-gpu[cuda,cudnn]`.
- Bumped `python_requires` to `>=3.11`; updated classifiers to 3.11/3.12/3.13.

### Tests (5 new modules, all CI-compatible)
| File                     | Coverage                                                                                                 |
| ------------------------ | -------------------------------------------------------------------------------------------------------- |
| `test_utils.py`          | `file_not_found`, `check_file`, `get_fname_woext`, `Rectangle`, enums, `setup_log`                       |
| `test_onnx_inference.py` | `is_gpu_available`, `ManagedOnnxSession` (CPU, fallback, unavailable), `OnnxInferencer` shapes + predict |
| `test_process_nd.py`     | `label_nd` (single class, no match, multi-component, multi-frame)                                        |
| `test_get_testdata.py`   | `get_modelfile` / `get_imagefile` + error paths                                                          |
| `test_sample_data.py`    | `make_sample_data` shape, dtype, value range                                                             |

- Removed `@pytest.mark.skipif(not ONNXRUNTIME_AVAILABLE)` from prediction
  tests so they execute in CI.
- GPU-only test (`test_gpu_available`) skipped in CI via `CI`/`GITHUB_ACTIONS`
  env vars.

### `README.md`
- New **CPU vs. GPU Inference** section with subsections: CPU (default), macOS,
  GPU (Windows/Linux), Troubleshooting GPU.
- Updated developer section with correct conda env filename.

### Other
- `.github/copilot-instructions.md`: repo-level context for GitHub Copilot.
- `env_napari_czann_segment.yml`: recreated as a clearly marked example.
- `tox.ini` / CI workflows: minor alignment.

## How to test

```bash
# CPU (all platforms)
pip install -e ".[testing]"
pytest -v

# GPU (Windows/Linux with CUDA 12.x)
pip install -e ".[gpu,testing]"
pytest -v   # test_gpu_available will run locally
```
