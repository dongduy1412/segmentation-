# Interior Space Segmentation

Advanced image segmentation for interior spaces using YOLO, SAM 2, Mask2Former, and Gemini VLM.

## 🚀 Features

- **YOLO + SAM 2**: Object detection with text prompts and refined masks
- **SAM 2 Everything**: Automatic mask generation with heuristic or VLM labeling
- **Mask2Former**: State-of-the-art panoptic segmentation for walls/floors/ceilings
- **Gemini VLM Integration**: AI-powered labeling using Google's Vision-Language Model

## 📦 Installation

```bash
# Initialize project
uv init --name segmentation
uv venv

# Install dependencies
uv pip install ultralytics torch torchvision Pillow click numpy opencv-python matplotlib
uv pip install sam2
uv pip install transformers timm
uv pip install google-generativeai
```

## 🎯 Usage

### 1. YOLO + SAM 2 (Object Detection)

Detect and segment specific objects using text prompts:

```bash
# Segment chairs
uv run python yolo_sam_segment.py --image_path test.jpg --prompt "chair"

# Segment couch
uv run python yolo_sam_segment.py --image_path test.jpg --prompt "couch" --no-display

# Custom confidence threshold
uv run python yolo_sam_segment.py --image_path test.jpg --prompt "chair" --conf 0.5
```

**Options:**
- `--image_path`: Input image path (required)
- `--prompt`: Text prompt for object detection (required)
- `--conf`: Confidence threshold (default: 0.25)
- `--model`: YOLO model path (default: yolov8n.pt)
- `--no-display`: Skip displaying result

**Best for:** Specific object segmentation (furniture, plants, etc.)

### 2. SAM 2 Everything (Automatic Segmentation)

Generate all possible masks and label them:

```bash
# Using heuristic labeling (default)
uv run python sam_everything.py --image_path test.jpg --no_display

# Using Gemini VLM for AI-powered labeling
uv run python sam_everything.py --image_path test.jpg --use_vlm --api_key YOUR_GEMINI_API_KEY

# Or set environment variable
export GEMINI_API_KEY=your_api_key_here
uv run python sam_everything.py --image_path test.jpg --use_vlm --no_display

# Save masks to JSON
uv run python sam_everything.py --image_path test.jpg --save_masks
```

**Options:**
- `--image_path`: Input image path (required)
- `--use_vlm`: Use Gemini VLM for labeling
- `--api_key`: Gemini API key (or set GEMINI_API_KEY env var)
- `--output`: Output image path (default: output_sam_everything.jpg)
- `--save_masks`: Save mask data as JSON
- `--min_area`: Minimum mask area (default: 100)
- `--no_display`: Skip displaying result

**Best for:** Exploratory analysis, detailed segmentation (70+ masks)

### 3. Mask2Former (Panoptic Segmentation) ⭐ **RECOMMENDED**

State-of-the-art segmentation for walls, floors, ceilings, and objects:

```bash
# Default (large model)
uv run python mask2former_segment.py --image_path test.jpg --no_display

# Use smaller/faster model
uv run python mask2former_segment.py --image_path test.jpg --model facebook/mask2former-swin-small-coco-panoptic

# Adjust confidence threshold
uv run python mask2former_segment.py --image_path test.jpg --threshold 0.7
```

**Options:**
- `--image_path`: Input image path (required)
- `--model`: Model size (tiny/small/base/large, default: large)
- `--output`: Output image path (default: output_mask2former.jpg)
- `--threshold`: Confidence threshold (default: 0.5)
- `--no_display`: Skip displaying result

**Available models:**
- `facebook/mask2former-swin-tiny-coco-panoptic` (fastest)
- `facebook/mask2former-swin-small-coco-panoptic`
- `facebook/mask2former-swin-base-coco-panoptic`
- `facebook/mask2former-swin-large-coco-panoptic` (most accurate)

**Best for:** Production use, accurate walls/floors/ceilings detection

## 🔑 Getting Gemini API Key

1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Use it with `--api_key` or set `GEMINI_API_KEY` environment variable

```bash
# Windows
set GEMINI_API_KEY=your_api_key_here

# Linux/Mac
export GEMINI_API_KEY=your_api_key_here
```

## 📊 Comparison

| Method | Walls | Floors | Ceiling | Objects | Speed | Accuracy |
|--------|-------|--------|---------|---------|-------|----------|
| YOLO + SAM 2 | ❌ | ❌ | ❌ | ✅✅ | Fast | High (for objects) |
| SAM Everything | ⚠️ | ⚠️ | ⚠️ | ✅ | Slow | Medium (heuristic) |
| SAM + Gemini VLM | ✅ | ✅ | ✅ | ✅ | Very Slow | High (AI-powered) |
| **Mask2Former** | ✅ | ✅ | ✅ | ✅ | **Medium** | **Very High** |

## 🎨 Output Examples

All scripts generate:
- Colored segmentation overlays
- Bounding boxes and labels
- Summary statistics
- Legend with color mappings

**Color scheme:**
- 🔵 **Blue shades**: Walls (different shades for left/right/back)
- 🟢 **Green**: Floors and rugs
- 🟣 **Pink/Purple**: Ceilings
- 🟠 **Orange**: Objects
- ⚪ **Gray**: Background/unknown

## 📁 Project Structure

```
segmentation/
├── yolo_sam_segment.py       # YOLO + SAM 2 object detection
├── sam_everything.py          # SAM 2 automatic + Gemini VLM
├── mask2former_segment.py     # Mask2Former panoptic segmentation
├── test.jpg                   # Sample interior image 1
├── test1.jpg                  # Sample interior image 2
├── sam2.1_b.pt               # SAM 2 model weights
├── yolov8n.pt                # YOLO model weights
└── README.md                  # This file
```

## 💡 Tips

1. **For walls/floors/ceilings**: Use Mask2Former (fastest and most accurate)
2. **For specific objects**: Use YOLO + SAM 2 with text prompts
3. **For exploration**: Use SAM Everything with Gemini VLM
4. **On CPU**: Use smaller models (tiny/small) for faster processing
5. **First run**: Models will be downloaded automatically (~1-2GB total)

## 🔧 Troubleshooting

**"SAM 2 checkpoint not found"**
- Ensure `sam2.1_b.pt` is in the project directory
- Download from [SAM 2 repository](https://github.com/facebookresearch/sam2)

**"YOLO model not found"**
- Ensure `yolov8n.pt` is in the project directory
- Will auto-download on first run

**"Gemini API error"**
- Check API key is valid
- Ensure you have API quota/credits
- Check internet connection

**Slow performance**
- Use GPU if available (CUDA)
- Use smaller models (tiny/small)
- Reduce image resolution

## 📝 Requirements

- Python 3.13+
- PyTorch with CPU/CUDA support
- 4GB+ RAM (8GB+ recommended)
- Internet connection (for first-time model downloads)

## 🎓 Citation

This project uses:
- [SAM 2](https://github.com/facebookresearch/sam2) by Meta AI
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [Mask2Former](https://github.com/facebookresearch/Mask2Former) by Meta AI
- [Google Gemini](https://ai.google.dev/) Vision API

## 📄 License

Educational and research purposes.
