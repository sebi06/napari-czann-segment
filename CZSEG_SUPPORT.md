# CZSEG Support Implementation Summary

## Overview
Extended `napari-czann-segment` to support both `.czann` and `.czseg` model file formats.

## Changes Made

### 1. New Parser Module: `czseg_parser.py`
Created `src/napari_czann_segment/czseg_parser.py` with:

- **`parse_czseg_xml()`**: Parses XML metadata from CZSEG files
  - Extracts: model ID, name, type, classes, border size, scaling flag
  - Handles **missing fields gracefully** (e.g., `TotalTileHeight`, `TotalTileWidth`)
  - Automatically infers tile dimensions from ONNX model when XML lacks them

- **`_infer_tile_size_from_onnx()`**: Inspects ONNX model input shape
  - Handles both NCHW and NHWC tensor formats
  - Used as fallback when XML is incomplete

- **`extract_czseg_model()`**: Main extraction function
  - Mirrors czmodel's `extract_czann_model()` API
  - Unpacks ZIP, locates `.xml` and `.model` files
  - Returns `(ModelMetadata, model_path)` tuple

### 2. Updated `dock_widget.py`
- Added import: `from napari_czann_segment.czseg_parser import extract_czseg_model`
- Updated `_read_model_metadata()` to:
  - Detect file extension (`.czann`, `.czmodel`, `.czseg`)
  - Route to appropriate parser
  - Log which format was detected

### 3. Updated `predict.py`
- Added import: `from napari_czann_segment.czseg_parser import extract_czseg_model`
- Updated `predict_ndarray()` to:
  - Detect file extension
  - Route to appropriate parser
  - Raise ValueError for unsupported formats

### 4. Updated File Filter in `dock_widget.py`
Changed filter from:
```python
model_extension = ["*.czann", "*.czmodel"]
```
To:
```python
model_extension = (
    "ZEISS model files (*.czann *.czmodel);;"
    "CZANN files (*.czann);;"
    "CZSEG files (*.czseg);;"
    "All files (*.*)"
)
```

### 5. Comprehensive Test Suite
Created `src/napari_czann_segment/_tests/test_czseg_parser.py`:

- `test_extract_czseg_with_tile_dimensions()`: Complete XML with tile dims
- `test_extract_czseg_without_tile_dimensions()`: XML missing tile dims (infers from ONNX)
- `test_czseg_vs_czann_compatibility()`: Validates CZANN/CZSEG produce compatible metadata
- `test_invalid_czseg_file()`: Error handling

**All 4 tests pass** ✓

## Key Features

### Robust Parsing
- Handles **complete** CZSEG XMLs (e.g., `PGC_20X_nucleus_detector.czseg`)
- Handles **incomplete** CZSEG XMLs (e.g., `260513_2025285_NMI-D_256_Uincep3_v1.czseg`)
- Falls back to sensible defaults (256×256) if ONNX inference also fails

### Metadata Compatibility
Both formats produce identical `ModelMetadata`:
```python
ModelMetadata(
    model_type=ModelType.SINGLE_CLASS_SEMANTIC_SEGMENTATION,
    input_shape=[H, W, C],
    output_shape=[H, W, num_classes],
    model_id="...",
    min_overlap=[overlap_h, overlap_w],
    classes=["background", "class1", ...],
    model_name="...",
    scaling=True/False
)
```

### XML Field Mapping
| CZSEG XML Field           | ModelMetadata Field | Notes                                        |
| ------------------------- | ------------------- | -------------------------------------------- |
| `<Id>`                    | `model_id`          |                                              |
| `<ModelName>`             | `model_name`        |                                              |
| `<TotalTileHeight>`       | `input_shape[0]`    | Inferred from ONNX if missing                |
| `<TotalTileWidth>`        | `input_shape[1]`    | Inferred from ONNX if missing                |
| `<BorderSize>`            | `min_overlap`       | `min_overlap = [borderSize*2, borderSize*2]` |
| `<TrainingClasses>`       | `classes`           | Extracted from `<Item Name="...">`           |
| `<ScaleInputsByBitdepth>` | `scaling`           | Boolean flag                                 |
| —                         | `output_shape`      | `[H, W, num_classes]`                        |

## Testing Results

### Unit Tests
```
pytest src/napari_czann_segment/_tests/test_czseg_parser.py -v
```
✓ 4 passed

### Files Tested
1. **`PGC_20X_nucleus_detector.czseg`**: Complete XML with all fields
2. **`260513_2025285_NMI-D_256_Uincep3_v1.czseg`**: Missing `TotalTileHeight/Width` (successfully inferred 256×256)

## Example Usage

### In napari Plugin
1. Open napari with the plugin
2. Use FileEdit widget to select a `.czseg` file
3. Metadata is automatically parsed and displayed
4. Proceed with segmentation as normal

### Programmatically
```python
from napari_czann_segment.czseg_parser import extract_czseg_model
from pathlib import Path
import tempfile

czseg_file = Path("model.czseg")
with tempfile.TemporaryDirectory() as temp_dir:
    metadata, model_path = extract_czseg_model(czseg_file, Path(temp_dir))
    print(metadata.classes)
    # Use model_path for ONNX inference
```

## Backward Compatibility
- All existing `.czann` and `.czmodel` workflows remain unchanged
- No breaking changes to API or user interface
- CZSEG support is additive only

## Dependencies
- No new dependencies added
- Uses existing: `onnxruntime`, `czmodel`, `xml.etree.ElementTree`
