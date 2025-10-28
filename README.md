# Image Segmentation

Advanced image segmentation application using state-of-the-art AI models.

## Features

### Method 1: YOLO + SAM 2 (Implemented)
- **YOLO (Ultralytics)**: Fast and accurate object detection
- **SAM 2 (Meta)**: Precise segmentation mask generation
- Combines speed of YOLO with precision of SAM

### Method 2: Grounding DINO + Mask2Former (Coming Soon)
- **Grounding DINO**: Text-prompt based object detection
- **Mask2Former**: Advanced semantic segmentation
- Enables natural language queries for segmentation

## Installation

This project uses `uv` for fast, reliable package management.

### Prerequisites
- Python 3.13+
- `uv` package manager

### Installing uv (if not already installed)

Choose one of the following methods:

**Option 1: pipx (Recommended)**
```bash
py -m pip install pipx
py -m pipx install uv
py -m pipx ensurepath
# Then restart terminal or refresh PATH
```

**Option 2: WinGet**
```bash
winget install --id=astral-sh.uv -e
```

**Option 3: PowerShell Installer**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

See `UV_MANAGEMENT.md` for detailed uv installation and management guide.

### Setup Project

```bash
# Navigate to project directory
cd segmentation

# Install dependencies (uv handles virtual environment automatically)
uv sync

# Or if PATH not updated yet:
py -m pipx run uv sync
```

## Usage

### Basic Usage

```bash
# Segment an image
uv run python main.py test.jpg
```

### Advanced Options

```bash
# Save output to specific file
uv run python main.py test.jpg --output result.jpg

# Higher confidence threshold (fewer, more confident detections)
uv run python main.py test.jpg --conf 0.5

# Detect specific object classes
# Example: person(0), chair(56), couch(57), dining table(60)
uv run python main.py test.jpg --classes 0 56 57 60

# Use larger YOLO model (more accurate, slower)
uv run python main.py test.jpg --yolo-model yolov8m.pt

# Disable SAM 2 refinement (faster but less precise masks)
uv run python main.py test.jpg --no-sam

# Don't display result window (just save)
uv run python main.py test.jpg --no-display

# Use specific device
uv run python main.py test.jpg --device cuda  # or cpu
```

### YOLO Model Options

- `yolov8n.pt` - Nano (fastest, least accurate)
- `yolov8s.pt` - Small
- `yolov8m.pt` - Medium (recommended)
- `yolov8l.pt` - Large
- `yolov8x.pt` - XLarge (most accurate, slowest)

## COCO Classes

Common object classes you can detect:

| ID | Class          | ID | Class        | ID | Class      |
|----|----------------|----|--------------|----|------------|
| 0  | person         | 56 | chair        | 60 | dining table |
| 1  | bicycle        | 57 | couch        | 61 | toilet     |
| 2  | car            | 58 | potted plant | 62 | tv         |
| 3  | motorcycle     | 59 | bed          | 63 | laptop     |

[Full COCO class list](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml)

## Output

The application generates:
1. **Segmented image** - With colored masks, bounding boxes, and labels
2. **Console summary** - Statistics of detected objects
3. **Visual comparison** - Side-by-side original vs segmented (if `--no-display` not used)

## Project Structure

```
segmentation/
├── src/
│   ├── __init__.py
│   ├── yolo_sam_segmenter.py    # YOLO + SAM 2 implementation
│   ├── advanced_segmenter.py    # Grounding DINO + Mask2Former (TODO)
│   └── utils.py                 # Utility functions
├── main.py                      # CLI interface
├── test.jpg                     # Sample image
├── pyproject.toml              # Project dependencies
└── README.md                   # This file
```

## How It Works

### YOLO + SAM 2 Pipeline

1. **Object Detection (YOLO)**
   - YOLO detects objects and provides bounding boxes
   - Fast and efficient for real-time applications
   - Identifies object classes and confidence scores

2. **Mask Refinement (SAM 2)**
   - SAM 2 takes YOLO's bounding boxes as prompts
   - Generates precise pixel-level segmentation masks
   - Handles complex boundaries and overlapping objects

3. **Visualization**
   - Overlays colored masks on original image
   - Draws bounding boxes and labels
   - Provides clear visual representation of segments

## Development

### Adding New Features

The codebase is designed for extensibility:

```python
from src.yolo_sam_segmenter import YOLOSAMSegmenter

# Initialize segmenter
segmenter = YOLOSAMSegmenter()

# Segment image
image, detections = segmenter.segment_image("image.jpg")

# Process results
for det in detections:
    print(f"{det['class_name']}: {det['confidence']:.2f}")
    if 'mask' in det:
        # Access segmentation mask
        mask = det['mask']
```

### Coming Soon: Grounding DINO + Mask2Former

This method will enable text-based prompts:

```bash
# Example (coming soon)
python main.py test.jpg --method advanced --prompt "segment all furniture"
```

## Troubleshooting

### CUDA Out of Memory
- Use smaller YOLO model: `--yolo-model yolov8n.pt`
- Use CPU: `--device cpu`
- Disable SAM: `--no-sam`

### Slow Performance
- Use faster model: `--yolo-model yolov8n.pt`
- Increase confidence threshold: `--conf 0.5`
- Disable SAM: `--no-sam`

### No Objects Detected
- Lower confidence threshold: `--conf 0.1`
- Try different YOLO model
- Check if objects are in COCO classes

## Credits

- **YOLO**: [Ultralytics](https://github.com/ultralytics/ultralytics)
- **SAM 2**: [Meta AI](https://github.com/facebookresearch/segment-anything-2)
- **Package Manager**: [uv](https://github.com/astral-sh/uv)

## License

This project is for educational and research purposes.