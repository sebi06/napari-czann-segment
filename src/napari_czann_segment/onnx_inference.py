# -*- coding: utf-8 -*-

#################################################################
# File        : onnx_inference.py
# Author      : sebi06, Team Enchilada
#
# Disclaimer: This code is purely experimental. Feel free to
# use it at your own risk.
#
#################################################################

from typing import Tuple, Optional, List, cast, Union, Dict, Any
from types import TracebackType
import logging
import os
import subprocess
import sys

import numpy as np

logger = logging.getLogger(__name__)
_CUDA_SESSION_USABLE_CACHE: Dict[str, bool] = {}
_CUDA_PROVIDER_OPTIONS = {
    "cudnn_conv_algo_search": "HEURISTIC",
    # Keep workspace bounded for 4 GB GPUs; larger workspaces can OOM on
    # EfficientNet-like convolution blocks even when the input tile is small.
    "cudnn_conv_use_max_workspace": "0",
}

# Ensure conda environment CUDA libraries are on the DLL search path (Windows).
# Conda may place CUDA DLLs in <env>/bin or <env>/Library/bin which are not
# always on PATH when launching from napari or other GUI entry points.
# Also covers pip-installed nvidia-* packages (e.g. nvidia-cublas-cu12) which
# place their DLLs under site-packages/nvidia/<pkg>/bin/.
if sys.platform == "win32":
    import site

    # conda CUDA directories
    _conda_prefix = os.environ.get("CONDA_PREFIX", "")
    if _conda_prefix:
        for _subdir in ("bin", os.path.join("Library", "bin")):
            _cuda_dir = os.path.join(_conda_prefix, _subdir)
            if os.path.isdir(_cuda_dir):
                os.add_dll_directory(_cuda_dir)
                if _cuda_dir not in os.environ.get("PATH", ""):
                    os.environ["PATH"] = _cuda_dir + os.pathsep + os.environ.get("PATH", "")

    # pip-installed nvidia-*-cu12 packages (site-packages/nvidia/<pkg>/bin/)
    for _sp in site.getsitepackages():
        _nvidia_dir = os.path.join(_sp, "nvidia")
        if os.path.isdir(_nvidia_dir):
            for _pkg in os.listdir(_nvidia_dir):
                _bin_dir = os.path.join(_nvidia_dir, _pkg, "bin")
                if os.path.isdir(_bin_dir):
                    os.add_dll_directory(_bin_dir)
                    if _bin_dir not in os.environ.get("PATH", ""):
                        os.environ["PATH"] = _bin_dir + os.pathsep + os.environ.get("PATH", "")

# Handle onnxruntime import gracefully for CI environments.
# On Windows CI (no GPU drivers), onnxruntime's native C extension may trigger
# a non-fatal access violation while probing for CUDA DLLs.  The process
# survives (the exception is handled internally via SEH), but Python's
# faulthandler — enabled by default in pytest — prints a scary
# "Windows fatal exception: access violation" traceback that causes CI to
# report the job as failed.  We temporarily disable faulthandler during the
# import so the benign SEH exception is silently swallowed.
import faulthandler as _fh

_fh_was_enabled = _fh.is_enabled()
try:
    if sys.platform == "win32":
        _fh.disable()
    import onnxruntime as rt

    # Verify the module actually has the required methods
    # (namespace packages may import successfully but be empty)
    if not hasattr(rt, "get_available_providers") or not hasattr(rt, "InferenceSession"):
        raise AttributeError("onnxruntime module is incomplete (missing required methods)")

    ONNXRUNTIME_AVAILABLE = True

    # Since onnxruntime-gpu >= 1.21, preload_dlls() can locate CUDA/cuDNN DLLs
    # from PyTorch, NVIDIA pip packages, or the system CUDA installation.
    # This must be called before creating any InferenceSession with CUDA.
    if hasattr(rt, "preload_dlls"):
        try:
            rt.preload_dlls()
            logger.debug("onnxruntime.preload_dlls() succeeded.")
        except Exception as e:
            logger.debug("onnxruntime.preload_dlls() failed (non-fatal): %s", e)

except (ImportError, AttributeError):
    # In CI environments, we might have DLL loading issues
    # Create a mock for basic functionality
    class MockOnnxRuntime:
        @staticmethod
        def InferenceSession(*args, **kwargs):
            raise ImportError("onnxruntime not available in CI environment")

    rt = MockOnnxRuntime()
    ONNXRUNTIME_AVAILABLE = False
finally:
    if _fh_was_enabled:
        _fh.enable()


def is_gpu_available() -> bool:
    """Check whether CUDA-based GPU inference is actually usable.

    Logs detailed diagnostic information about the GPU availability
    to help users troubleshoot their environment.

    Returns True only if onnxruntime-gpu is installed, CUDAExecutionProvider
    is listed, and a minimal CUDA session can be created.
    """
    if not ONNXRUNTIME_AVAILABLE:
        logger.info("GPU check: onnxruntime is not installed.")
        return False

    version = getattr(rt, "__version__", "unknown")
    logger.info("GPU check: onnxruntime %s installed.", version)

    try:
        providers = rt.get_available_providers()
    except Exception as e:
        logger.warning("GPU check: Failed to query available providers: %s", e)
        return False

    logger.info("GPU check: Available providers: %s", providers)

    if "CUDAExecutionProvider" not in providers:
        logger.info(
            "GPU check: CUDAExecutionProvider is NOT available. "
            "Only CPU inference will be used. "
            "To enable GPU, install onnxruntime-gpu: pip install napari-czann-segment[gpu]"
        )
        return False

    logger.info(
        "GPU check: CUDAExecutionProvider is listed. "
        "A per-model CUDA session preflight will run before GPU inference."
    )
    return True


def _is_cuda_session_usable(model_path: str) -> bool:
    """Check CUDA session creation in a child process.

    Some CUDA/cuDNN/cuBLAS library mismatches abort the Python process from
    native code instead of raising a catchable Python exception.  Running the
    CUDA session probe in a subprocess lets the napari process fall back to CPU
    safely when that happens.
    """

    cached = _CUDA_SESSION_USABLE_CACHE.get(model_path)
    if cached is not None:
        return cached

    code = """
import sys
import numpy as np
import onnxruntime as rt

model_path = sys.argv[1]
providers = [
    (
        "CUDAExecutionProvider",
        {
            "cudnn_conv_algo_search": "HEURISTIC",
            "cudnn_conv_use_max_workspace": "0",
        },
    ),
    "CPUExecutionProvider",
]

if hasattr(rt, "preload_dlls"):
    try:
        rt.preload_dlls()
    except Exception:
        pass

sess = rt.InferenceSession(model_path, providers=providers)
active = sess.get_providers()
if "CUDAExecutionProvider" not in active:
    raise RuntimeError(f"CUDAExecutionProvider was not activated. Active providers: {active}")

input_info = sess.get_inputs()[0]
input_shape = [1 if not isinstance(dim, int) else dim for dim in input_info.shape]
input_shape[0] = 1
dummy = np.zeros(input_shape, dtype=np.float32)
sess.run([sess.get_outputs()[0].name], {input_info.name: dummy})
"""

    env = os.environ.copy()
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    try:
        result = subprocess.run(
            [sys.executable, "-B", "-c", code, model_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
            env=env,
        )
    except Exception as e:
        logger.warning("CUDA preflight failed before inference: %s. Falling back to CPU.", e)
        _CUDA_SESSION_USABLE_CACHE[model_path] = False
        return False

    if result.returncode != 0:
        stderr = result.stderr.strip()
        logger.warning(
            "CUDA preflight failed for model %s with return code %s. "
            "Falling back to CPU to avoid crashing napari. stderr: %s",
            model_path,
            result.returncode,
            stderr[-1000:],
        )
        _CUDA_SESSION_USABLE_CACHE[model_path] = False
        return False

    _CUDA_SESSION_USABLE_CACHE[model_path] = True
    return True


class ManagedOnnxSession:
    """Context manager for managing ONNX inference sessions.

    This context manager allows you to create and manage ONNX inference sessions.
    It provides a convenient way to handle the creation and deletion of the session.

    Usage:
        with ManagedOnnxSession(model_path, providers) as session:
            # Use the session for inference

    Attributes:
        _model_path (str): The path to the ONNX model on disk.
        providers (Optional[List[Union[str, Tuple[str, Dict]]]]): The names of the provider classes
            to be used to retrieve an execution device.

    Methods:
        __enter__(): Creates an ONNX inference session and returns it.
        __exit__(exc_type, exc_val, exc_tb): Deletes the ONNX inference session.

    """

    def __init__(
        self,
        model_path: str,
        providers: Optional[List[Union[str, Tuple[str, Dict]]]] = None,
    ) -> None:
        """Creates an instance of the ManagedOnnxSession context manager.

        Args:
            model_path (str): The path to the ONNX model on disk.
            providers (Optional[List[Union[str, Tuple[str, Dict]]]]): The names of the provider classes
                to be used to retrieve an execution device.

        """
        self._model_path = model_path
        self.providers = providers

        # Check if onnxruntime is available during initialization
        if not ONNXRUNTIME_AVAILABLE:
            raise ImportError(
                "onnxruntime is not available in this environment. " "This is likely due to missing dependencies in CI."
            )

    def __enter__(self) -> Any:
        """Creates an ONNX inference session and returns it.

        Falls back to CPUExecutionProvider if the requested providers (e.g. CUDA) fail.

        Returns:
            rt.InferenceSession: The created ONNX inference session.

        """
        try:
            self._session = rt.InferenceSession(self._model_path, providers=self.providers)
        except Exception as e:
            cpu_only = ["CPUExecutionProvider"]
            if self.providers != cpu_only:
                logger.warning(
                    "Failed to create ONNX session with providers %s: %s. " "Falling back to CPUExecutionProvider.",
                    self.providers,
                    e,
                )
                self._session = rt.InferenceSession(self._model_path, providers=cpu_only)
            else:
                raise
        return self._session

    def __exit__(
        self,
        exc_type: BaseException,
        exc_val: BaseException,
        exc_tb: TracebackType,
    ) -> None:
        """Deletes the ONNX inference session.

        Args:
            exc_type (BaseException): The type of the exception raised, if any.
            exc_val (BaseException): The exception raised, if any.
            exc_tb (TracebackType): The traceback of the exception raised, if any.

        """
        del self._session


class OnnxInferencer:
    """Inferencer class to load and evaluate models in ONNX format."""

    def __init__(self, model_path: str, batch_size: int = 8) -> None:
        """Creates an instance of a ONNX inferencer.

        Arguments:
            model_path: The path to the model on disk.
            batch_size: Number of images to process in a single batch (default: 8).
                       Increase for faster inference on GPUs with more memory.
                       Set to 1 to revert to sequential processing (uses less memory).
        """
        super().__init__()
        self._model_path = model_path
        self._batch_size = batch_size

    @property
    def batch_size(self) -> int:
        """Number of images processed per GPU forward pass."""
        return self._batch_size

    def predict(self, x: List[np.ndarray], use_gpu: bool = False) -> List[np.ndarray]:
        """Evaluates the underlying model with the given input _data.

        Arguments:
            x: The input _data to evaluate the model with.
            use_gpu: Allow execution on GPU (True) or enforce CPU execution (False).

        Returns:
            The prediction for the given input _data.
        """

        def _predict_batch(_x: List[np.ndarray], use_gpu: bool = True, batch_size: int = 1) -> List[np.ndarray]:
            """Run prediction on a batch of images.

            Processes images in batches to improve GPU utilization and reduce
            inference overhead compared to sequential per-image processing.

            Arguments:
                _x: The batch of images to be predicted.
                use_gpu: Allow execution on GPU (True) or enforce CPU execution (False).
                batch_size: Number of images per ONNX forward pass.

            Returns:
                 The predictions for the given batch of images.
            """

            # Use cuDNN's heuristic algorithm selection. It avoids many slow
            # fallback Conv paths without the startup and memory cost of the
            # exhaustive benchmark search.
            with ManagedOnnxSession(
                self._model_path,
                providers=(
                    [  # "TensorrtExecutionProvider",
                        (
                            "CUDAExecutionProvider",
                            _CUDA_PROVIDER_OPTIONS,
                        ),
                        "CPUExecutionProvider",
                    ]
                    if use_gpu
                    else ["CPUExecutionProvider"]
                ),
            ) as sess:

                input_name = sess.get_inputs()[0].name
                output_name = sess.get_outputs()[0].name

                # Check if the model supports dynamic batching.
                # ONNX models exported with a fixed batch size (e.g. 1) cannot
                # receive a tensor with N > 1 on the batch axis.  When the batch
                # dimension is an integer we must process one image at a time.
                model_batch_dim = sess.get_inputs()[0].shape[0]
                effective_batch = 1 if isinstance(model_batch_dim, int) else max(1, batch_size)
                if effective_batch == 1 and batch_size > 1:
                    logger.debug(
                        "Model has fixed batch size (%s); falling back to " "single-image inference.",
                        model_batch_dim,
                    )

                prediction_list = []

                for batch_start in range(0, len(_x), effective_batch):
                    batch_end = min(batch_start + effective_batch, len(_x))
                    batch_images = _x[batch_start:batch_end]

                    # Stack images into a single batch tensor
                    # Shape: (N, H, W, C) where N is the effective batch size
                    batch_tensor = np.stack(batch_images, axis=0).astype(np.float32)

                    # Run inference on entire batch at once (single forward pass)
                    batch_result = sess.run([output_name], {input_name: batch_tensor})[0]

                    # Unstack results back to per-image format
                    for i in range(batch_end - batch_start):
                        prediction_list.append(batch_result[i])

                return prediction_list

        if use_gpu and not _is_cuda_session_usable(self._model_path):
            use_gpu = False

        if not use_gpu:
            return _predict_batch(x, use_gpu=False, batch_size=self._batch_size)

        gpu_batch_size = max(1, self._batch_size)
        while gpu_batch_size >= 1:
            try:
                return _predict_batch(x, use_gpu=True, batch_size=gpu_batch_size)
            except Exception as e:
                if gpu_batch_size == 1:
                    logger.warning(
                        "GPU inference failed even with batch size 1: %s. "
                        "Falling back to CPUExecutionProvider for this prediction.",
                        e,
                    )
                    return _predict_batch(x, use_gpu=False, batch_size=1)

                next_batch_size = max(1, gpu_batch_size // 2)
                logger.warning(
                    "GPU inference failed with batch size %s: %s. "
                    "Retrying with batch size %s.",
                    gpu_batch_size,
                    e,
                    next_batch_size,
                )
                gpu_batch_size = next_batch_size

    def get_input_shape(self) -> Tuple[int, int, int, int]:
        """Determines the input shape expected by the loaded model.

        Using CPUExecutionProvider straight from the first run to not try-except for CUDAExecutionProvider - fast op.

        Returns:
            The expected input shape.
        """
        with ManagedOnnxSession(self._model_path, providers=["CPUExecutionProvider"]) as sess:
            input_shape = tuple(elem if isinstance(elem, int) else None for elem in sess.get_inputs()[0].shape)
            if len(input_shape) != 4:
                raise ValueError(
                    f"The input shape of the model must have four dimensions. Found dimensions: {input_shape}"
                )
            return cast(Tuple[int, int, int, int], input_shape)

    def get_output_shape(
        self,
    ) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[int]]:
        """Determines the output shape of the loaded model.

        Using CPUExecutionProvider straight from the first run to not try-except for CUDAExecutionProvider - fast op.

        Returns:
            The output shape of the model.
        """
        with ManagedOnnxSession(self._model_path, providers=["CPUExecutionProvider"]) as sess:
            output_shape = tuple(elem if isinstance(elem, int) else None for elem in sess.get_outputs()[0].shape)
            if len(output_shape) != 4:
                raise ValueError(
                    f"The output shape of the model must have four dimensions. Found dimensions: {output_shape}"
                )
            return cast(
                Tuple[Optional[int], Optional[int], Optional[int], Optional[int]],
                output_shape,
            )
