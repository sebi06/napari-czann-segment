# ONNX Inference Comparison: napari-czann-segment vs. SegmentationService

## Summary
napari-czann-segment and SegmentationService both use ONNX Runtime for inference. While they produce **identical results for Test 1** (PGC model), **Test 2** (UIncep3 model) fails in napari but works in SegmentationService.

**Updated conclusion (June 26, 2026): the primary root cause is intensity scaling, not tiling and not RGB/BGR alone.** The UIncep3 `.czseg` metadata contains `<ScaleInputsByBitdepth>false</ScaleInputsByBitdepth>`, but the napari widget forced `do_rescale=True` for all semantic segmentation models. This incorrectly compressed `uint8` input values from `0..255` to `0..1` for a model that expects unscaled values.

RGB/BGR channel order may still matter after the correct intensity scale is used: the model declares `PixelType="Bgr24"`. However, earlier tests showing RGB/BGR conversion had no effect were misleading because they were performed while the input was incorrectly rescaled.

The CZI loading path was also clarified: images are loaded through the `napari-czitools` plugin, which uses `czitools`/`pylibCZIrw` under the hood. For `Image 12 small.czi`, `napari-czitools` returns an xarray layer with dimensions `('S', 'T', 'Z', 'Y', 'X', 'A')` and shape `(1, 1, 1, 4205, 5238, 3)`, so the RGB/BGR sample axis is last and XY tiling is valid for this plugin path.

---

## Updated Findings and Fix (June 26, 2026)

### Confirmed Model and Image Metadata

For `260513_2025285_NMI-D_256_Uincep3_v1.czseg`:

- ONNX input shape: `(None, 256, 256, 3)`
- ONNX output shape: `(None, 256, 256, 5)`
- Parsed model input shape: `[256, 256, 3]`
- Parsed model output shape: `[256, 256, 5]`
- Classes: `Nitrides`, `Sulfides`, `Oxides`, `Artefacts`, `Steel`
- XML channel metadata: `<Item PixelType="Bgr24" />`
- XML color handling: `<ColorHandling>SplitRgb</ColorHandling>`
- XML scaling metadata: `<ScaleInputsByBitdepth>false</ScaleInputsByBitdepth>`
- Border size: `50`, parsed as `min_overlap=[100, 100]`

For `Image 12 small.czi` loaded by `napari-czitools`:

- Returned layer type: `xarray.DataArray`
- Dimensions: `('S', 'T', 'Z', 'Y', 'X', 'A')`
- Shape: `(1, 1, 1, 4205, 5238, 3)`
- Dtype: `uint8`
- Value range observed: `0..216`

### Primary Root Cause: Ignored Scaling Metadata

The `.czseg` parser already reads `ScaleInputsByBitdepth` into `model_metadata.scaling`, but the widget ignored it:

```python
predict_ndarray(
    ...,
    do_rescale=True,
)
```

For the UIncep3 model this is wrong because `ScaleInputsByBitdepth=false`. The model expects raw `uint8`-scale values, but napari-czann-segment divided the input by `255`.

On a `256 x 256 x 3` crop:

- `do_rescale=False`, RGB/BGR off: classes `1, 2, 4, 5`
- `do_rescale=True`, RGB/BGR off: mostly classes `4, 5`
- Difference between unscaled and scaled results: `58,135 / 65,536` pixels
- RGB/BGR conversion changed results when `do_rescale=False`
- RGB/BGR conversion had `0` pixel effect when `do_rescale=True`

This explains why RGB/BGR testing initially appeared irrelevant: the model output had already collapsed because the intensity scale was wrong.

### Secondary Bug: Plain `(Y, X, 3)` Arrays

`predict_ndarray()` previously detected channel-last arrays only when `len(img.shape) > 3`. That worked for `napari-czitools` data shaped `(1, 1, 1, Y, X, 3)`, but failed for plain RGB/BGR arrays shaped `(Y, X, 3)`.

For a crop shaped `(256, 256, 3)`, the wrapper treated `Y` as a stack axis and sent an invalid `(X, 3)` slice to `predict_tiles2d()`, causing:

```text
ValueError: Channel mismatch: tile has 1 channel(s), but model expects 3.
```

`predict_tiles2d()` itself handled `(Y, X, 3)` correctly. The bug was in the multidimensional wrapper.

### Implemented Fix

The fix has two parts:

1. `predict_ndarray(..., do_rescale=None)` now uses model metadata:

```python
if do_rescale is None:
    do_rescale = getattr(modelmd, "scaling", True)
```

2. The widget now passes the selected model's scaling metadata instead of forcing rescaling:

```python
do_rescale=getattr(self.model_metadata, "scaling", True)
```

3. Channel-last detection now happens after model metadata is loaded and checks the model's expected channel count, so both `(Y, X, 3)` and `(..., Y, X, 3)` are handled correctly.

### Current Diagnosis

For the real `napari-czitools` loading path, CZTILE is not the primary issue. The data arrives as `(..., Y, X, A)` with the sample/channel axis last, and tiles are extracted across XY only.

The primary issue was preprocessing: the UIncep3 model's `ScaleInputsByBitdepth=false` metadata was ignored. After fixing scaling, RGB/BGR channel order should be evaluated separately because the model still declares `Bgr24`.

---

## Historical Investigation Notes (Superseded)

The following sections document the original investigation path. They are retained for context, but the current diagnosis above supersedes the early RGB/BGR-only hypothesis.

## Key Differences Found

### 1. **Color Channel Format (RGB vs. BGR)** ⚠️ Original Hypothesis, Not Primary Root Cause

#### Model Metadata: Both Models Expect BGR24 Format

From the extracted model XML (`260513_2025285_NMI-D_256_Uincep3_v1.xml`):
```xml
<ColorHandling>SplitRgb</ColorHandling>
<Channels>
  <Item PixelType="Bgr24" />
</Channels>
```

Both Test 1 (PGC) and Test 2 (UIncep3) models declare they expect **BGR24** pixel format with SplitRgb color handling.

#### napari-czann-segment: No RGB/BGR Conversion

In `predict.py`, the `_prep()` function does:
```python
def _prep(t: np.ndarray) -> np.ndarray:
    if do_rescale:
        t = t.astype(np.float32) / np.iinfo(img2d.dtype).max  # Normalize to [0, 1]
    if t.ndim == 2:
        t = t[..., np.newaxis]  # Add channel dimension
    return t
    # NO COLOR CHANNEL REORDERING!
```

**napari loads images in RGB format** (standard for image viewers and libraries), but never converts them to BGR before inference.

#### SegmentationService: Also No Explicit BGR Conversion in DNN Path

In `classifier.py` lines 336-356:
```python
def predict(self, x: np.ndarray) -> np.ndarray:
    # ... no BGR/RGB conversion ...
    batch_elem = batch_elem[np.newaxis]  # Add batch dimension
    batch_elem = batch_elem.astype(np.float32)  # Convert to float
    result = sess.run([output_name], input_dict)
    # NO COLOR CHANNEL REORDERING!
```

SegmentationService also doesn't explicitly convert RGB→BGR for DNN inference (that conversion is only in the VGG19 feature extractors, which are a separate path).

### The Critical Difference: Image Source Format

**Hypothesis:** The actual input images to napari and SegmentationService may come from different sources or be preprocessed differently:

- **napari:** Loads images through image viewer/libraries that default to RGB format
  - CZI images are read and displayed in RGB
  - Model expects BGR → **Channel mismatch**
  - Result: R and B channels are swapped → Artifacts

- **SegmentationService:** May load/preprocess images already in BGR format OR the input test data is already BGR
  - Images are in the correct BGR format
  - Model expects BGR → **Correct format**
  - Result: Works correctly

### Why This Matters for Test 2

For deep neural networks like UIncep3 (especially Inception architectures), the **first convolutional layer is extremely sensitive to color information**:

1. **Feature detection depends on color order:**
   - Inception blocks use parallel color-specific feature extractors
   - RGB vs. BGR swap causes R and B channel features to be swapped
   - Early layers extract edges, textures, colors that are specific to channel order

2. **Swapped channels = Completely different features:**
   - Early layers see wrong color cues → Extract wrong feature maps
   - Deep layers make decisions based on corrupted features
   - Result: Incorrect segmentation with visible artifacts

3. **Why Test 1 (PGC) still works:**
   - PGC is a simpler model, possibly less color-sensitive
   - May work with grayscale or be more robust to channel order
   - Or possibly the input image is already grayscale/single-channel

---

### 2. **Batch Size Processing** (Not the root cause)

#### napari-czann-segment (supports dynamic batching)
```python
batch_tensor = np.stack(batch_images, axis=0).astype(np.float32)  # Shape: (N, H, W, C)
batch_result = sess.run([output_name], {input_name: batch_tensor})[0]
```

#### SegmentationService (always batch size 1)
```python
for batch_elem in x:
    batch_elem = batch_elem[np.newaxis]  # Add batch dimension → (1, H, W, C)
    result = sess.run([output_name], input_dict)[0]
    prediction_list.append(result)
```

**Status:** Not the root cause. Both systems process images through the model successfully. The color channel issue would affect both equally regardless of batch size.

---

### 3. **Data Preprocessing**

#### napari-czann-segment (`predict.py` lines 145-159)
```python
def _prep(t: np.ndarray) -> np.ndarray:
    if isinstance(t, da.Array):
        t = t.compute()
    if do_rescale:
        t = t.astype(np.float32) / np.iinfo(img2d.dtype).max  # Normalize to [0, 1]
    else:
        t = t.astype(np.float32)
    if t.ndim == 2:
        t = t[..., np.newaxis]  # Add channel dimension if missing
    if t.shape[-1] != input_channels:
        raise ValueError(...)
    return t
```

#### SegmentationService (via `DnnDataManager`)
```python
# load_dataset(..., scale_by_bitdepth=self.scale_inputs_by_bitdepth)
# Also normalizes to [0, 1] based on pixel type
```

**Both use identical preprocessing:**
- Scales pixel values: `[0, max_value] → [0, 1]`
- Adds channel dimension if missing
- ✅ **Preprocessing is NOT the issue**

---

## Test Results Analysis

### Test 1: PGC_20X_nucleus_detector.czseg ✅ Both work **IDENTICALLY**
- **Results:** napari and SegmentationService produce **identical outputs**
- **Model type:** Simpler segmentation model
- **Model metadata:** Also declares `PixelType="Bgr24"`
- **Why it works despite RGB/BGR mismatch:** 
  - May operate primarily on grayscale/intensity data
  - Or color sensitivity is low, making RGB↔BGR swap less impactful
  - Or input data happens to be grayscale
- **Observation:** Both batch-size-1 and dynamic batching produce the same results
- **Conclusion:** The basic ONNX inference pipeline works correctly in both systems

### Test 2: 260513_2025285_NMI-D_256_Uincep3_v1.czseg ❌ napari fails, ✅ SegmentationService works
- **Results:** napari produces artifacts/wrong results, SegmentationService works correctly
- **Model type:** UNet-Inception v3 variant (very deep, color-sensitive architecture)
- **Model metadata:** Explicitly declares `PixelType="Bgr24"`
- **Key issue:** napari provides RGB images, model expects BGR
- **Why it fails:**
  - Inception blocks are highly color-sensitive
  - First convolutional layer uses color-specific filters
  - R and B channels swapped → Wrong features extracted
  - Downstream layers use corrupted features → Wrong segmentation
- **Historical conclusion, superseded:** The issue appeared to be **color channel order (RGB vs. BGR)** rather than batch processing. Later testing showed incorrect intensity scaling was the primary cause.

## Root Cause Analysis

### Key Evidence
1. **Both models declare BGR24 input format** in their metadata
   - `<Channels><Item PixelType="Bgr24" /></Channels>`
   - `<ColorHandling>SplitRgb</ColorHandling>`

2. **napari loads images in RGB format** (standard for image viewers/libraries)
   - No BGR conversion happens in `_prep()`
   - CZI images displayed by napari are in RGB
   - **Result:** Model receives R and B channels swapped

3. **Test 1 (PGC) is robust** to this channel swap
   - Simpler model, less color-sensitive
   - May operate on grayscale or single-channel data
   - Works despite RGB/BGR mismatch

4. **Test 2 (UIncep3) fails** due to RGB/BGR mismatch
   - Inception architecture is highly color-sensitive
   - First convolutional layers extract color-specific features
   - R↔B channel swap corrupts early feature extraction
   - Downstream layers receive wrong feature maps
   - Result: Incorrect segmentation with artifacts

### Why SegmentationService Works
The input images to SegmentationService are likely:
1. Already in BGR format when loaded (proper preprocessing upstream)
2. Or the test uses pre-converted BGR data
3. Result: Model receives correct channel order

### Diagnosis
The UIncep3 model trained on **BGR** (standard for OpenCV, ZEISS imaging), but napari provides **RGB** data. This 1:1 mismatch in color channel order causes the model to extract completely wrong features from images.

---

## Debugging - Investigation (June 26, 2026)

### CRITICAL FINDING: RGB/BGR Conversion Is NOT the Solution

**Testing Results:**
- ✅ Image correctly loaded: (1, 1, 1, 4205, 5238, 3) where 3 = RGB/BGR pixel values
- ✅ Checkbox implementation: WORKING (parameter correctly toggles True/False)
- ✅ Conversion code: EXECUTING (logs confirm "RGB to BGR CONVERSION IS ENABLED")
- ❌ **Conversion effect: NO IMPACT** (results identical whether ON or OFF)

**Historical conclusion, refined:** RGB/BGR conversion alone did not change results in this test. Later testing showed this was because the input was incorrectly rescaled first; RGB/BGR may still matter after scaling is corrected.

### Root Cause: Identified Later as Scaling Metadata

At this point in the investigation the root cause was not yet known. The later finding was that `ScaleInputsByBitdepth=false` was ignored. The candidate list at this stage was:
1. **CZI loading/parsing difference** - napari vs SegmentationService read differently
2. **Preprocessing step missing** - SegmentationService does something napari doesn't
3. **Model input layout** - possible NCHW vs NHWC mismatch
4. **Tiling/stitching strategy** - different boundary handling or padding
5. **Model version difference** - wrong model file or outdated version
6. **Data arrangement** - pixel value order or normalization difference

### Next Investigation Steps

1. **Compare CZI file loading:**
   - How does SegmentationService load the same CZI file?
   - Does it produce different pixel values or layout?

2. **Verify model files:**
   - Are napari and SegmentationService using identical model files?
   - Check MD5 checksums of the .onnx files

3. **Test preprocessing:**
   - Check if SegmentationService does additional preprocessing steps
   - Compare pixel value ranges after loading but before inference

4. **Visual comparison:**
   - Load same image in both systems side-by-side
   - Document exact differences in output
   - Note patterns in artifacts (where they appear, what they look like)

---

## Implemented Solution

### Primary Fix: Honor `ScaleInputsByBitdepth`

The `.czseg` parser already exposes the scaling metadata as `model_metadata.scaling`. Prediction now uses that metadata by default:

```python
def predict_ndarray(..., do_rescale: Optional[bool] = None, ...):
    ...
    if do_rescale is None:
        do_rescale = getattr(modelmd, "scaling", True)
```

The napari widget no longer forces scaling for semantic segmentation:

```python
modeldata, seg_complete = predict_ndarray(
    self.czann_file,
    img_layer.data,
    border=self.min_overlap_ui,
    use_gpu=self.use_gpu,
    do_rescale=getattr(self.model_metadata, "scaling", True),
    tiling_method=self.tiling_method,
    merge_window=self.merge_method,
    batch_size=self.batch_size,
)
```

For `260513_2025285_NMI-D_256_Uincep3_v1.czseg`, this means `do_rescale=False`, matching the XML metadata.

### Secondary Fix: Model-Aware Channel-Axis Detection

The wrapper now loads model metadata before deciding whether the last axis is a channel/sample axis. It checks the model's expected input channel count:

```python
input_channels = modelmd.input_shape[-1]
has_channel_dim = len(img.shape) >= 3 and img.shape[-1] == input_channels and (
    input_channels > 1 or len(img.shape) == 3
)
```

This preserves correct behavior for `napari-czitools` arrays shaped `(1, 1, 1, Y, X, 3)` and fixes plain RGB/BGR crops shaped `(Y, X, 3)`.

### RGB/BGR Status

RGB/BGR conversion is no longer considered the primary fix. The model still declares `PixelType="Bgr24"`, so channel order may still need explicit handling after intensity scaling is correct. The earlier test result where conversion had no effect was caused by the wrong intensity scale.

---

## Updated Code Comparison Table

| Aspect                 | napari                  | SegmentationService                 | Impact on Test 2 |
| ---------------------- | ----------------------- | ----------------------------------- | ---------------- |
| **Input scaling**      | Previously forced `[0, max] -> [0, 1]`; now uses model metadata | Uses `ScaleInputsByBitdepth` metadata | ⚠️ **Primary issue** |
| **Color Channels**     | `napari-czitools` returns sample axis last as `A=3` | Service path still needs comparison | ⚠️ Still possible after scaling |
| **Channel Conversion** | Optional/debug path only | None in DNN path                    | ⚠️ Secondary |
| **Batch Size**         | Dynamic (1 to N)        | Fixed (always 1)                    | ℹ️ Minor          |
| **Normalization**      | Now honors `model_metadata.scaling` | Uses model metadata                 | ✅ Aligned        |
| **cuDNN Search**       | EXHAUSTIVE              | Default                             | ℹ️ Minor          |
| **GPU Fallback**       | Yes                     | Yes                                 | ✅ Identical      |
---

## Expected Test Results After Implemented Fix

After honoring `ScaleInputsByBitdepth` and fixing channel-axis detection:

```
Test 1: PGC_20X_nucleus_detector.czseg
- Before: Works
- After: Should continue to work because its scaling metadata is true

Test 2: 260513_2025285_NMI-D_256_Uincep3_v1.czseg
- Before: Wrong/artifact-prone because input was incorrectly rescaled
- After: Should use raw uint8-scale input, matching ScaleInputsByBitdepth=false
```

RGB/BGR channel order should be re-evaluated after this fix using correctly scaled input.

---

## References

- [ONNX Runtime Execution Providers Documentation](https://onnxruntime.ai/docs/execution-providers/)
- [cuDNN Algorithm Selection & Performance](https://docs.nvidia.com/deeplearning/cudnn/developer-guide/index.html)
- [napari-czann-segment onnx_inference.py](./src/napari_czann_segment/onnx_inference.py)
- [SegmentationService classifier.py](../../../RMS_SegmentationService_Container/SegmentationService/segmentationservice/core/classifier.py)
