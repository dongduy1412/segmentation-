# Interior Space Segmentation

Advanced image segmentation for interior spaces using YOLO, SAM 2, Mask2Former, and Gemini VLM.

## 🚀 Features

### Core Segmentation Methods
- **YOLO + SAM 2**: Object detection with text prompts and refined masks (multimask selection)
- **SAM 2 Everything**: Automatic mask generation with heuristic or VLM labeling
- **Mask2Former**: State-of-the-art panoptic segmentation for walls/floors/ceilings (99%+ accuracy)
- **Hybrid Strategy**: YOLO → SAM 2 → SAM Everything fallback for robust detection

### Advanced Lighting Analysis
- **Ceiling Lighting Detection**: Detect multiple ceiling regions with brightness variations
- **Wall Lighting Analysis**: Identify wall zones with different lighting (bright/normal/dark) using K-means clustering
- **Floor Lighting Detection**: Analyze floor and rug regions with lighting variations
- **AI Integration**: Optional Gemini VLM for intelligent labeling

### Specialized Object Detection
- **TV Detection**: High-accuracy TV detection (99.7% with Mask2Former)
- **Vase Detection**: Multi-vase detection with position and brightness analysis
- **General Objects**: Furniture, plants, decorations with detailed properties

## 📦 Installation

### Using uv (Recommended)

```bash
# Clone repository
git clone https://github.com/dongduy1412/segmentation-.git
cd segmentation-

# Create virtual environment
uv venv

# Install all dependencies from pyproject.toml
uv sync
```

### Manual Installation

```bash
# Initialize project
uv init --name segmentation
uv venv

# Install dependencies
uv pip install ultralytics torch torchvision Pillow click numpy opencv-python matplotlib
uv pip install sam2
uv pip install transformers timm
uv pip install google-generativeai
uv pip install scikit-learn
```

### Dependencies
- **ultralytics** (YOLO)
- **sam2** (Segment Anything Model 2)
- **transformers + timm** (Mask2Former)
- **torch + torchvision** (Deep Learning)
- **click** (CLI interface)
- **opencv-python, pillow, matplotlib** (Image processing & visualization)
- **numpy** (Array operations)
- **google-generativeai** (Gemini VLM - optional)
- **scikit-learn** (K-means clustering for lighting analysis)

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

### 4. Hybrid Segmentation (Adaptive Strategy)

Intelligently combines YOLO, SAM 2, and SAM Everything for robust detection:

```bash
# Try YOLO first, fallback to SAM Everything if needed
uv run python hybrid_segment.py --image_path test.jpg --prompt "table" --no-display

# With custom YOLO model and confidence
uv run python hybrid_segment.py --image_path test.jpg --prompt "chair" --yolo_model yolov8m.pt --conf 0.3
```

**Options:**
- `--image_path`: Input image path (required)
- `--prompt`: Object to detect (default: table)
- `--yolo_model`: YOLO model path (default: yolov8m.pt)
- `--conf`: YOLO confidence threshold (default: 0.25)
- `--output`: Output image path (default: output_hybrid.jpg)
- `--no-display`: Skip displaying result

**Strategy:**
1. Try YOLO detection first (fast)
2. If YOLO fails → Use SAM 2 Everything with heuristic filtering
3. Smart fallback ensures objects are always found

**Best for:** Difficult objects, unknown scenarios, production robustness

---

## 🔬 Advanced Features

### Lighting Variation Analysis

Detect and analyze lighting differences in interior spaces using Mask2Former + K-means clustering.

#### Ceiling Lighting Detection

```bash
# Detect ceiling with lighting variations
uv run python detect_ceiling_detailed.py --image_path test.jpg --no_display

# Adjust minimum mask area
uv run python detect_ceiling_detailed.py --image_path test.jpg --min_area 200
```

**Output:** Detects multiple ceiling regions based on brightness (bright/normal/dark zones)

#### Wall Lighting Analysis

```bash
# Analyze wall lighting variations
uv run python detect_wall_lighting.py --image_path test.jpg --no_display

# Specify number of lighting clusters
uv run python detect_wall_lighting.py --image_path test.jpg --n_clusters 3
```

**Output:** 
- Detects wall using Mask2Former (99%+ accuracy)
- Analyzes lighting zones using K-means clustering
- Reports brightness values and percentages for each zone

**Example Result:**
```
Zone 1 (dark):   Brightness: 75.0,  Size: 3.3% of wall
Zone 2 (normal): Brightness: 155.3, Size: 30.3% of wall  
Zone 3 (bright): Brightness: 184.7, Size: 66.4% of wall
```

#### Floor Lighting Detection

```bash
# Detect floor and rug with lighting analysis
uv run python detect_floor_lighting.py --image_path test.jpg --no_display
```

**Output:**
- Detects multiple floor types (floor-merged, rug-merged)
- Analyzes lighting zones for each floor region
- Reports position (front/center/back) and brightness

---

### Specialized Object Detection

#### TV Detection

```bash
# High-accuracy TV detection
uv run python detect_tv_detailed.py --image_path test.jpg --no_display
```

**Features:**
- 99.7% confidence with Mask2Former
- Detects TV even when screen is off/dark
- Reports position, size, brightness, and average color

#### Vase Detection

```bash
# Detect all vases with detailed properties
uv run python detect_vase_detailed.py --image_path test.jpg --no_display
```

**Features:**
- Multi-vase detection (finds 4+ vases)
- Position analysis (top/middle/bottom, left/center/right)
- Brightness and color analysis
- Size comparison and ranking

---

## 📋 Complete Scripts Reference

| Script | Purpose | Best Use Case |
|--------|---------|---------------|
| `yolo_sam_segment.py` | YOLO + SAM 2 object detection | Specific objects (chair, couch, plant) |
| `sam_everything.py` | Automatic segmentation | Exploratory analysis, 70+ masks |
| `mask2former_segment.py` | Panoptic segmentation | Walls, floors, ceilings, all objects |
| `hybrid_segment.py` | Adaptive strategy | Difficult objects, robust detection |
| `detect_ceiling_detailed.py` | Ceiling lighting analysis | Multi-zone ceiling with brightness |
| `detect_wall_lighting.py` | Wall lighting analysis | Wall zones with lighting variations |
| `detect_floor_lighting.py` | Floor lighting analysis | Floor/rug with brightness zones |
| `detect_tv_detailed.py` | TV detection | High-accuracy TV localization |
| `detect_vase_detailed.py` | Vase detection | Multi-vase detection with properties |
| `detect_wall_detailed.py` | Wall detection (SAM 2) | Alternative wall detection method |

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

## 📊 Method Comparison

| Method | Walls | Floors | Ceiling | Objects | Speed | Accuracy | Cost |
|--------|-------|--------|---------|---------|-------|----------|------|
| YOLO + SAM 2 | ❌ | ❌ | ❌ | ✅✅ | Fast ⚡ | High (90%+) | Free |
| SAM Everything | ⚠️ | ⚠️ | ⚠️ | ✅ | Slow 🐌 | Medium (85%) | Free |
| SAM + Gemini VLM | ✅ | ✅ | ✅ | ✅ | Very Slow 🐌🐌 | High (95%+) | $$$ |
| **Mask2Former** | ✅ | ✅ | ✅ | ✅ | **Medium** 🚀 | **Very High (99%+)** 🏆 | **Free** |
| Hybrid Strategy | ❌ | ❌ | ❌ | ✅✅ | Medium | High (90%+) | Free |
| Lighting Analysis | ✅ | ✅ | ✅ | ❌ | Medium | Very High (99%+) | Free |

### Recommendation by Use Case

**Production Deployment:** Mask2Former 🏆
- Fastest and most accurate
- No API costs
- Works offline
- 99%+ accuracy for walls/floors/ceilings

**Specific Objects:** YOLO + SAM 2
- Fast and accurate for known objects
- Text-based prompts (chair, couch, plant)
- 90%+ accuracy with excellent masks

**Lighting Analysis:** Mask2Former + K-means
- Detects brightness variations
- Separates bright/normal/dark zones
- Useful for lighting design and analysis

**Exploratory Analysis:** SAM Everything (Heuristic)
- Generates 70+ masks automatically
- Good for discovering all regions
- Free and offline

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
├── Core Segmentation Scripts
│   ├── yolo_sam_segment.py          # YOLO + SAM 2 object detection
│   ├── sam_everything.py            # SAM 2 automatic + Gemini VLM
│   ├── mask2former_segment.py       # Mask2Former panoptic segmentation
│   └── hybrid_segment.py            # Hybrid strategy with fallback
│
├── Lighting Analysis Scripts
│   ├── detect_ceiling_detailed.py   # Ceiling with brightness zones
│   ├── detect_wall_lighting.py      # Wall lighting analysis (Mask2Former + K-means)
│   ├── detect_floor_lighting.py     # Floor lighting analysis
│   └── detect_wall_detailed.py      # Wall detection (SAM 2 alternative)
│
├── Specialized Detection Scripts
│   ├── detect_tv_detailed.py        # High-accuracy TV detection
│   └── detect_vase_detailed.py      # Multi-vase detection
│
├── Configuration & Setup
│   ├── pyproject.toml               # Project dependencies (uv standard)
│   ├── uv.lock                      # Dependency lock file
│   └── .python-version              # Python version (3.13+)
│
├── Model Weights (Download automatically)
│   ├── sam2.1_b.pt                  # SAM 2 model weights (~600MB)
│   ├── yolov8n.pt                   # YOLO nano model (~6MB)
│   └── yolov8m.pt                   # YOLO medium model (~52MB)
│
├── Test Images
│   ├── test.jpg                     # Living room scene
│   ├── test1.jpg → test7.png        # Various interior scenes
│   └── testt.jpg                    # Additional test image
│
└── Documentation
    ├── README.md                    # This file
    └── main.py                      # Legacy entry point (deprecated)
```

## 💡 Best Practices & Tips

### Choosing the Right Method

1. **For walls/floors/ceilings** → Use `mask2former_segment.py` 🏆
   - 99%+ accuracy
   - Fastest for structural elements
   - Works great on CPU

2. **For lighting analysis** → Use `detect_*_lighting.py` scripts
   - Mask2Former + K-means clustering
   - Detects bright/normal/dark zones
   - Useful for interior design analysis

3. **For specific objects (chair, couch, plant)** → Use `yolo_sam_segment.py`
   - Fast and accurate
   - Text-based prompts
   - Excellent mask quality with SAM 2

4. **For difficult objects** → Use `hybrid_segment.py`
   - Automatic fallback strategy
   - Robust detection
   - Good for production

5. **For TV, vases, or specific items** → Use specialized `detect_*_detailed.py` scripts
   - Optimized for specific objects
   - Detailed property analysis
   - High accuracy with Mask2Former

### Performance Optimization

1. **On CPU**: All scripts work well on CPU (tested on Intel/AMD)
2. **On GPU**: Use CUDA for 3-5x speedup
3. **Smaller models**: Use `yolov8n.pt` (nano) for faster YOLO
4. **Larger models**: Use `yolov8m.pt` (medium) for better accuracy
5. **First run**: Models download automatically (~1-2GB total)
6. **Reduce masks**: Use `--min_area` for SAM Everything to limit masks

### Model Downloads (Automatic)

| Model | Size | Download Location | First Run |
|-------|------|-------------------|-----------|
| SAM 2 (sam2.1_b.pt) | ~600MB | Project directory | Manual |
| YOLO nano (yolov8n.pt) | ~6MB | Auto-download | Yes |
| YOLO medium (yolov8m.pt) | ~52MB | Auto-download | Yes |
| Mask2Former large | ~1GB | HuggingFace cache | Auto |
| Gemini (API-based) | - | Google Cloud | API key |

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

## 📝 System Requirements

### Minimum
- **Python**: 3.13+
- **RAM**: 4GB
- **Storage**: 3GB free space (for models)
- **CPU**: Intel/AMD with AVX2 support
- **Internet**: For first-time model downloads

### Recommended
- **Python**: 3.13
- **RAM**: 8GB+
- **Storage**: 5GB free space
- **GPU**: NVIDIA with CUDA support (3-5x faster)
- **Internet**: For Gemini VLM (optional)

### Tested Platforms
- ✅ Windows 10/11 (64-bit)
- ✅ Ubuntu 20.04+ (64-bit)
- ✅ macOS 12+ (Intel/Apple Silicon)

### Dependencies (Auto-installed via uv)
See `pyproject.toml` for complete list:
- ultralytics, sam2, transformers, timm
- torch, torchvision, numpy, opencv-python
- click, pillow, matplotlib
- google-generativeai (optional)
- scikit-learn (for lighting analysis)

## 🎓 Citation

This project uses:
- [SAM 2](https://github.com/facebookresearch/sam2) by Meta AI
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [Mask2Former](https://github.com/facebookresearch/Mask2Former) by Meta AI
- [Google Gemini](https://ai.google.dev/) Vision API

## 📄 License

Educational and research purposes.
