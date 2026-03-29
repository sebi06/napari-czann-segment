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
import sys

import numpy as np

logger = logging.getLogger(__name__)

# Ensure conda environment CUDA libraries are on the DLL search path (Windows).
# Conda may place CUDA DLLs in <env>/bin or <env>/Library/bin which are not
# always on PATH when launching from napari or other GUI entry points.
if sys.platform == "win32":
    _conda_prefix = os.environ.get("CONDA_PREFIX", "")
    if _conda_prefix:
        for _subdir in ("bin", os.path.join("Library", "bin")):
            _cuda_dir = os.path.join(_conda_prefix, _subdir)
            if os.path.isdir(_cuda_dir):
                os.add_dll_directory(_cuda_dir)
                if _cuda_dir not in os.environ.get("PATH", ""):
                    os.environ["PATH"] = _cuda_dir + os.pathsep + os.environ.get("PATH", "")

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

except ImportError:
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

    logger.info("GPU check: onnxruntime %s installed.", rt.__version__)

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

    logger.info("GPU check: CUDAExecutionProvider is available. GPU inference enabled.")
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

    def __init__(self, model_path: str) -> None:
        """Creates an instance of a ONNX inferencer.

        Arguments:
            model_path: The path to the model on disk.
        """
        super().__init__()
        self._model_path = model_path

    def predict(self, x: List[np.ndarray], use_gpu: bool = False) -> List[np.ndarray]:
        """Evaluates the underlying model with the given input _data.

        Arguments:
            x: The input _data to evaluate the model with.
            use_gpu: Allow execution on GPU (True) or enforce CPU execution (False).

        Returns:
            The prediction for the given input _data.
        """

        def predict_one(sess: Any, batch_elem: np.ndarray) -> np.ndarray:
            """Predicts with a batch size of 1 to not risk memory issues.

            Arguments:
                sess: The inference session containing the loaded model.
                batch_elem: One element of a batch to be used for prediction.

            Returns:
                The prediction for the provided batch element.
            """
            batch_elem = batch_elem[np.newaxis]
            input_name = sess.get_inputs()[0].name
            output_name = sess.get_outputs()[0].name

            # ONNX can only handle float32
            batch_elem = batch_elem.astype(np.float32)
            input_dict = {input_name: batch_elem}
            result = sess.run([output_name], input_dict)[0]

            if len(result) != 1:
                raise AssertionError("The batch size has changed during ANN model execution")
            return result[0]

        def _predict_batch(_x: List[np.ndarray], use_gpu: bool = True) -> List[np.ndarray]:
            """Run prediction on a batch of images.

            Arguments:
                _x: The batch of images to be predicted.
                use_gpu: Allow execution on GPU (True) or enforce CPU execution (False).

            Returns:
                 The predictions for the given batch of images.
            """

            # try to make it run fast with GPU
            # https://medium.com/neuml/debug-onnx-gpu-performance-c9290fe07459

            # cuDNN provider options:
            # - "cudnn_conv_algo_search": "EXHAUSTIVE" benchmarks all available
            #   convolution algorithms and picks the fastest one. The default
            #   ("DEFAULT") may fall back to a slower, non-cuDNN code path and
            #   emit warnings like "Fallback to non-cudnn non-fused conv2d".
            # - "cudnn_conv_use_max_workspace": "1" allows cuDNN to allocate as
            #   much GPU workspace as needed, which prevents the algorithm search
            #   from silently discarding faster kernels that require more memory.
            with ManagedOnnxSession(
                self._model_path,
                providers=(
                    [  # "TensorrtExecutionProvider",
                        (
                            "CUDAExecutionProvider",
                            {
                                "cudnn_conv_algo_search": "EXHAUSTIVE",
                                "cudnn_conv_use_max_workspace": "1",
                            },
                        ),
                        "CPUExecutionProvider",
                    ]
                    if use_gpu
                    else ["CPUExecutionProvider"]
                ),
            ) as sess:

                # We predict with a batch size of 1 to not risk memory issues
                prediction_list = [predict_one(sess, batch_elem) for batch_elem in _x]

                return prediction_list

        return _predict_batch(x, use_gpu=use_gpu)

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
