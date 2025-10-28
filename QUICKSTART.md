# Quick Start Guide

Get started with image segmentation in 5 minutes!

## Installation

```bash
# Navigate to project directory
cd D:\segmentation

# Install dependencies (uv handles everything automatically)
py -m uv sync
```

## Basic Usage

### 1. Segment Any Image
```bash
py -m uv run python main.py test.jpg
```
This will:
- Detect all objects in the image
- Generate precise segmentation masks
- Save result as `test_segmented.jpg`
- Display side-by-side comparison

### 2. Save to Custom Location
```bash
py -m uv run python main.py test.jpg --output my_result.jpg
```

### 3. Faster Processing (No Display)
```bash
py -m uv run python main.py test.jpg --no-display
```

## Common Use Cases

### Segment Furniture Only
```bash
# Detect: chairs, couches, plants, tables
py -m uv run python main.py test.jpg --classes 56 57 58 60
```

### Segment People
```bash
# Detect only people
py -m uv run python main.py test.jpg --classes 0
```

### High Confidence Only
```bash
# Only show very confident detections
py -m uv run python main.py test.jpg --conf 0.5
```

### Fast Mode (Skip SAM 2)
```bash
# YOLO only - much faster but less precise masks
py -m uv run python main.py test.jpg --no-sam
```

### Better Accuracy (Larger Model)
```bash
# Use medium-sized YOLO for better accuracy
py -m uv run python main.py test.jpg --yolo-model yolov8m.pt
```

## Python API

### Basic Example
```python
from src.yolo_sam_segmenter import YOLOSAMSegmenter
from src.utils import save_image

# Initialize
segmenter = YOLOSAMSegmenter()

# Segment
image, detections = segmenter.segment_image("test.jpg")

# Print results
segmenter.print_summary(detections)

# Visualize
result = segmenter.visualize_results(image, detections)
save_image(result, "output.jpg")
```

### Process Multiple Images
```python
from pathlib import Path
from src.yolo_sam_segmenter import YOLOSAMSegmenter

segmenter = YOLOSAMSegmenter()

for img_path in Path(".").glob("*.jpg"):
    print(f"Processing {img_path}...")
    image, detections = segmenter.segment_image(str(img_path))
    result = segmenter.visualize_results(image, detections)
    
    output_name = f"{img_path.stem}_segmented.jpg"
    save_image(result, output_name)
```

### Filter by Confidence
```python
segmenter = YOLOSAMSegmenter()
image, detections = segmenter.segment_image(
    "test.jpg",
    conf_threshold=0.5  # Only 50%+ confidence
)

# Further filter manually
high_conf = [d for d in detections if d['confidence'] > 0.7]
print(f"Found {len(high_conf)} high-confidence objects")
```

## Troubleshooting

### "Out of memory" Error
```bash
# Use smaller model
py -m uv run python main.py test.jpg --yolo-model yolov8n.pt --no-sam

# Or force CPU
py -m uv run python main.py test.jpg --device cpu
```

### "No objects detected"
```bash
# Lower confidence threshold
py -m uv run python main.py test.jpg --conf 0.1
```

### Slow Performance
```bash
# Use nano model + skip SAM
py -m uv run python main.py test.jpg --yolo-model yolov8n.pt --no-sam
```

## Next Steps

1. **Explore Examples**
   ```bash
   py -m uv run python example_usage.py
   ```

2. **Read Full Documentation**
   - See `README.md` for complete features
   - See `COCO_CLASSES.md` for all 80 object classes
   - See `PROJECT_SUMMARY.md` for technical details

3. **Customize**
   - Modify `src/yolo_sam_segmenter.py` for custom processing
   - Add new visualization styles in `src/utils.py`
   - Extend with your own models

## Tips & Tricks

### 1. Speed Up Processing
- Use `--no-sam` for 3-5x speedup
- Use `yolov8n.pt` (nano) model
- Use `--device cuda` if you have GPU
- Increase `--conf` threshold

### 2. Improve Accuracy
- Use `yolov8m.pt` or larger models
- Lower `--conf` threshold (e.g., 0.1)
- Keep SAM 2 enabled (default)
- Process higher resolution images

### 3. Working with Specific Objects
- Check `COCO_CLASSES.md` for class IDs
- Use `--classes` to filter specific objects
- Combine multiple classes for room-specific detection

### 4. Batch Processing
```bash
# Windows PowerShell
Get-ChildItem *.jpg | ForEach-Object {
    py -m uv run python main.py $_.Name --no-display
}
```

### 5. Custom Visualization
```python
# In your script
result = segmenter.visualize_results(
    image,
    detections,
    show_boxes=False,    # Hide bounding boxes
    show_masks=True,     # Show masks
    show_labels=False    # Hide labels
)
```

## Class ID Quick Reference

Common objects:
- 0: person
- 56: chair
- 57: couch
- 58: potted plant
- 59: bed
- 60: dining table
- 62: tv
- 63: laptop

See `COCO_CLASSES.md` for complete list of 80 classes.

## Getting Help

```bash
# Show all CLI options
py -m uv run python main.py --help

# Check Python version
py --version

# Check installed packages
py -m uv pip list
```

---

**Ready to segment!** 🚀

Start with: `py -m uv run python main.py test.jpg`
