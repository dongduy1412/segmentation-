"""
SAM 2 Everything Mode + VLM Labeling
Automatically segments everything in an image and uses VLM to label components
(walls, floors, ceilings, objects)
"""

import click
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import cv2
import json
import os


def label_mask_with_vlm(image_crop, mask_area, gemini_model=None):
    """
    Use Vision-Language Model (Gemini) to label a mask region
    
    Args:
        image_crop: Cropped image region (numpy array)
        mask_area: Area of the mask (for filtering small regions)
        gemini_model: Initialized Gemini model (optional)
    
    Returns:
        Label string: 'wall', 'floor', 'ceiling', 'object', or 'unknown'
    """
    if gemini_model is None:
        return 'unknown'
    
    try:
        # Convert numpy array to PIL Image
        if isinstance(image_crop, np.ndarray):
            image_pil = Image.fromarray(cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB))
        else:
            image_pil = image_crop
        
        # Prepare prompt for Gemini
        prompt = """Analyze this image region and classify it into ONE of these categories:
- wall: Any wall surface (painted, brick, concrete, wood panel, tile, etc.)
- floor: Any floor surface (wood, tile, marble, stone, carpet, rug, etc.)
- ceiling: Any ceiling surface
- object: Any furniture, decoration, or movable object
- unknown: If you cannot determine

Reply with ONLY ONE WORD: wall, floor, ceiling, object, or unknown.
Do not explain, just give the single word classification."""

        # Call Gemini Vision API
        response = gemini_model.generate_content([prompt, image_pil])
        
        # Parse response
        label = response.text.strip().lower()
        
        # Validate response
        valid_labels = ['wall', 'floor', 'ceiling', 'object', 'unknown']
        if label in valid_labels:
            return label
        
        # Try to extract valid label from response
        for valid_label in valid_labels:
            if valid_label in label:
                return valid_label
        
        return 'unknown'
        
    except Exception as e:
        # Fallback to unknown if API fails
        # Debug: print error for first few failures
        import sys
        if not hasattr(label_mask_with_vlm, '_error_count'):
            label_mask_with_vlm._error_count = 0
        if label_mask_with_vlm._error_count < 3:
            print(f"[DEBUG] Gemini API error: {str(e)}", file=sys.stderr)
            label_mask_with_vlm._error_count += 1
        return 'unknown'


def simple_heuristic_labeling(mask, image_shape):
    """
    Simple heuristic to label masks based on position
    (Temporary until VLM is integrated)
    
    Args:
        mask: Binary mask
        image_shape: (height, width) of image
    
    Returns:
        Label string
    """
    h, w = image_shape[:2]
    
    # Get mask properties
    coords = np.where(mask > 0)
    if len(coords[0]) == 0:
        return 'unknown'
    
    y_coords, x_coords = coords
    center_y = np.mean(y_coords)
    center_x = np.mean(x_coords)
    
    min_y, max_y = np.min(y_coords), np.max(y_coords)
    min_x, max_x = np.min(x_coords), np.max(x_coords)
    
    height_span = (max_y - min_y) / h
    width_span = (max_x - min_x) / w
    area_ratio = np.sum(mask) / (h * w)
    
    # Heuristics:
    # - Ceiling: Top 30% of image, wide horizontal span
    # - Floor: Bottom 40% of image, wide horizontal span
    # - Wall: Vertical span > 50%, positioned left/right or center-back
    # - Object: Everything else
    
    # Ceiling detection
    if center_y < h * 0.3 and width_span > 0.5 and area_ratio > 0.05:
        return 'ceiling'
    
    # Floor detection
    if center_y > h * 0.6 and width_span > 0.5 and area_ratio > 0.1:
        return 'floor'
    
    # Wall detection (large vertical regions)
    if height_span > 0.5 and area_ratio > 0.05:
        # Left wall
        if center_x < w * 0.25:
            return 'wall_left'
        # Right wall
        elif center_x > w * 0.75:
            return 'wall_right'
        # Back wall (center)
        elif width_span > 0.3:
            return 'wall_back'
    
    # Default to object
    return 'object'


@click.command()
@click.option('--image_path', required=True, type=click.Path(exists=True), help='Path to input image')
@click.option('--use_vlm', is_flag=True, help='Use Gemini VLM for labeling (requires --api_key)')
@click.option('--api_key', type=str, default=None, help='Gemini API key (or set GEMINI_API_KEY env var)')
@click.option('--output', default='output_sam_everything.jpg', help='Output image path')
@click.option('--save_masks', is_flag=True, help='Save individual masks as JSON')
@click.option('--min_area', default=100, type=int, help='Minimum mask area to consider')
@click.option('--no_display', is_flag=True, help='Skip displaying the result')
def main(image_path, use_vlm, api_key, output, save_masks, min_area, no_display):
    """Segment everything in image using SAM 2 and label with VLM"""
    
    click.echo("="*60)
    click.echo("SAM 2 Everything Mode + VLM Labeling")
    click.echo("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    click.echo(f"Using device: {device}")
    
    # Initialize Gemini VLM if requested
    gemini_model = None
    if use_vlm:
        try:
            import google.generativeai as genai
            
            # Get API key from argument or environment
            if api_key is None:
                api_key = os.environ.get('GEMINI_API_KEY')
            
            if api_key is None:
                click.echo("\nWarning: --use_vlm specified but no API key provided")
                click.echo("Set GEMINI_API_KEY environment variable or use --api_key")
                click.echo("Falling back to heuristic labeling...")
                use_vlm = False
            else:
                click.echo("\nInitializing Gemini Vision API...")
                genai.configure(api_key=api_key)
                # Use latest Gemini 2.5 Flash model
                gemini_model = genai.GenerativeModel('gemini-2.5-flash')
                click.echo("[OK] Gemini VLM ready (model: gemini-2.5-flash)")
        except ImportError:
            click.echo("\nWarning: google-generativeai not installed")
            click.echo("Install with: uv pip install google-generativeai")
            click.echo("Falling back to heuristic labeling...")
            use_vlm = False
        except Exception as e:
            click.echo(f"\nWarning: Could not initialize Gemini: {str(e)}")
            click.echo("Falling back to heuristic labeling...")
            use_vlm = False
    
    # Load image
    click.echo(f"\nLoading image: {image_path}")
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]
    
    # Load SAM 2 model for automatic mask generation
    click.echo("\nLoading SAM 2 Automatic Mask Generator...")
    try:
        from sam2.build_sam import build_sam2
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        import sam2
        
        sam2_checkpoint = "sam2.1_b.pt"
        sam2_base = Path(sam2.__file__).parent
        model_cfg = str(sam2_base / "configs" / "sam2.1" / "sam2.1_hiera_b+.yaml")
        
        if not Path(sam2_checkpoint).exists():
            click.echo(f"Error: SAM 2 checkpoint not found at {sam2_checkpoint}")
            click.echo("Please download sam2.1_b.pt model file")
            return
        
        sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
        
        # Create automatic mask generator
        mask_generator = SAM2AutomaticMaskGenerator(
            model=sam2_model,
            points_per_side=32,
            pred_iou_thresh=0.86,
            stability_score_thresh=0.92,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=min_area,
        )
        
        click.echo("[OK] SAM 2 model loaded successfully")
        
    except Exception as e:
        click.echo(f"Error loading SAM 2: {str(e)}")
        return
    
    # Generate masks
    click.echo("\nGenerating all possible masks...")
    click.echo("This may take a while...")
    
    masks = mask_generator.generate(image_rgb)
    
    click.echo(f"[OK] Generated {len(masks)} masks")
    
    # Sort masks by area (largest first)
    masks = sorted(masks, key=lambda x: x['area'], reverse=True)
    
    # Label each mask
    click.echo(f"\nLabeling masks ({'VLM' if use_vlm else 'heuristic'} mode)...")
    if use_vlm:
        click.echo(f"Note: VLM labeling is slow (~5-10s per mask)")
        click.echo(f"Consider using --min_area to reduce number of masks")
    
    labeled_masks = []
    label_counts = {}
    
    for idx, mask_data in enumerate(masks):
        if use_vlm and idx % 5 == 0:
            click.echo(f"  Processing mask {idx+1}/{len(masks)}...")
        mask = mask_data['segmentation']
        area = mask_data['area']
        
        if use_vlm and gemini_model is not None:
            # Get bounding box and crop image
            bbox = mask_data['bbox']  # [x, y, w, h]
            x, y, w_box, h_box = map(int, bbox)
            image_crop = image_rgb[y:y+h_box, x:x+w_box]
            
            # Use Gemini VLM for labeling
            label = label_mask_with_vlm(image_crop, area, gemini_model)
        else:
            # Use simple heuristics
            label = simple_heuristic_labeling(mask, image.shape)
        
        labeled_masks.append({
            'mask': mask,
            'label': label,
            'area': area,
            'bbox': mask_data['bbox']
        })
        
        label_counts[label] = label_counts.get(label, 0) + 1
    
    # Print summary
    click.echo(f"\n{'='*40}")
    click.echo("Segmentation Summary:")
    click.echo(f"{'='*40}")
    for label, count in sorted(label_counts.items()):
        click.echo(f"  {label:15s}: {count:3d} regions")
    click.echo(f"{'='*40}")
    
    # Color mapping for visualization
    color_map = {
        'ceiling': (135, 206, 235),      # Sky blue
        'wall_back': (70, 130, 180),     # Steel blue
        'wall_left': (100, 149, 237),    # Cornflower blue
        'wall_right': (65, 105, 225),    # Royal blue
        'floor': (60, 179, 113),         # Medium sea green
        'object': (255, 165, 0),         # Orange
        'unknown': (128, 128, 128),      # Gray
    }
    
    # Create output visualization
    click.echo("\nCreating visualization...")
    output_image = image.copy()
    overlay = np.zeros_like(image, dtype=np.uint8)
    
    for mask_info in labeled_masks:
        mask = mask_info['mask']
        label = mask_info['label']
        color = color_map.get(label, (200, 200, 200))
        
        # Apply colored mask
        overlay[mask] = color
    
    # Blend with original image
    output_image = cv2.addWeighted(image, 0.4, overlay, 0.6, 0)
    
    # Add legend
    legend_height = 200
    legend = np.ones((legend_height, w, 3), dtype=np.uint8) * 255
    
    y_pos = 30
    for label, color in color_map.items():
        if label in label_counts:
            # Draw color box
            cv2.rectangle(legend, (20, y_pos-15), (50, y_pos), color, -1)
            cv2.rectangle(legend, (20, y_pos-15), (50, y_pos), (0, 0, 0), 1)
            
            # Draw text
            text = f"{label}: {label_counts[label]} regions"
            cv2.putText(legend, text, (60, y_pos-2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            y_pos += 25
    
    # Combine image and legend
    final_output = np.vstack([output_image, legend])
    
    # Save output
    cv2.imwrite(output, final_output)
    click.echo(f"\n[OK] Output saved to: {output}")
    
    # Save masks as JSON if requested
    if save_masks:
        masks_data = []
        for mask_info in labeled_masks:
            masks_data.append({
                'label': mask_info['label'],
                'area': int(mask_info['area']),
                'bbox': [float(x) for x in mask_info['bbox']]
            })
        
        json_path = output.replace('.jpg', '_masks.json')
        with open(json_path, 'w') as f:
            json.dump(masks_data, f, indent=2)
        click.echo(f"[OK] Masks data saved to: {json_path}")
    
    # Display result
    if not no_display:
        click.echo("\nDisplaying result...")
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(1, 2, figsize=(14, 7))
            
            axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            axes[0].set_title('Original Image')
            axes[0].axis('off')
            
            axes[1].imshow(cv2.cvtColor(final_output, cv2.COLOR_BGR2RGB))
            axes[1].set_title(f'Segmented ({len(masks)} masks)')
            axes[1].axis('off')
            
            plt.tight_layout()
            plt.show()
        except Exception as e:
            click.echo(f"Could not display image: {str(e)}")
    
    click.echo("\n[DONE]")


if __name__ == '__main__':
    main()
