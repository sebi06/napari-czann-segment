from types import SimpleNamespace

import numpy as np

from napari_czann_segment import predict


class DummyInferencer:
    def __init__(self, model_path, batch_size):
        self.model_path = model_path
        self.batch_size = batch_size


def test_predict_ndarray_uses_model_scaling_and_keeps_yxc(monkeypatch):
    model_md = SimpleNamespace(
        input_shape=[4, 4, 3],
        min_overlap=[1, 1],
        scaling=False,
    )
    calls = []

    def fake_extract(path, target_dir):
        return model_md, target_dir / "model.onnx"

    def fake_predict_tiles2d(
        img2d,
        model_md,
        inferencer,
        min_border_width,
        do_rescale,
        use_gpu,
        tiling_method,
        merge_window,
        convert_rgb_to_bgr,
    ):
        calls.append((img2d.shape, do_rescale))
        return np.zeros(img2d.shape[:2], dtype=img2d.dtype)

    monkeypatch.setattr(predict, "extract_czseg_model", fake_extract)
    monkeypatch.setattr(predict, "OnnxInferencer", DummyInferencer)
    monkeypatch.setattr(predict, "predict_tiles2d", fake_predict_tiles2d)

    img = np.zeros((8, 9, 3), dtype=np.uint8)

    _, seg = predict.predict_ndarray("model.czseg", img)

    assert seg.shape == (8, 9)
    assert calls == [((8, 9, 3), False)]


def test_predict_ndarray_preserves_stacked_yxc_output_shape(monkeypatch):
    model_md = SimpleNamespace(
        input_shape=[4, 4, 3],
        min_overlap=[1, 1],
        scaling=True,
    )
    calls = []

    def fake_extract(path, target_dir):
        return model_md, target_dir / "model.onnx"

    def fake_predict_tiles2d(
        img2d,
        model_md,
        inferencer,
        min_border_width,
        do_rescale,
        use_gpu,
        tiling_method,
        merge_window,
        convert_rgb_to_bgr,
    ):
        calls.append((img2d.shape, do_rescale))
        return np.zeros(img2d.shape[:2], dtype=img2d.dtype)

    monkeypatch.setattr(predict, "extract_czseg_model", fake_extract)
    monkeypatch.setattr(predict, "OnnxInferencer", DummyInferencer)
    monkeypatch.setattr(predict, "predict_tiles2d", fake_predict_tiles2d)

    img = np.zeros((1, 2, 8, 9, 3), dtype=np.uint8)

    _, seg = predict.predict_ndarray("model.czseg", img)

    assert seg.shape == (1, 2, 8, 9)
    assert calls == [
        ((8, 9, 3), True),
        ((8, 9, 3), True),
    ]
