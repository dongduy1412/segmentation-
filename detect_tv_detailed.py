"""
Detailed TV Detection
Uses Mask2Former to detect TV with detailed visualization
"""

import click
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import cv2
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor


@click.command()
@click.option('--image_path', required=True, type=click.Path(exists=True), help='Path to input image')
@click.option('--output', default='output_tv_detailed.jpg', help='Output image path')
@click.option('--threshold', default=0.5, type=float, help='Confidence threshold')
@click.option('--no_display', is_flag=True, help='Skip displaying the result')
def main(image_path, output, threshold, no_display):
    """Detailed TV detection"""
    
    click.echo("="*60)
    click.echo("Detailed TV Detection")
    click.echo("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    click.echo(f"Using device: {device}")
    
    # Load model
    click.echo("\nLoading Mask2Former model...")
    model_name = "facebook/mask2former-swin-large-coco-panoptic"
    processor = Mask2FormerImageProcessor.from_pretrained(model_name)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(model_name)
    click.echo("[OK] Model loaded")
    
    # Load image
    click.echo(f"\nLoading image: {image_path}")
    image_pil = Image.open(image_path).convert("RGB")
    image_cv = cv2.imread(image_path)
    h, w = image_cv.shape[:2]
    
    # Run inference
    click.echo("\nRunning segmentation...")
    inputs = processor(images=image_pil, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    results = processor.post_process_panoptic_segmentation(
        outputs,
        target_sizes=[image_pil.size[::-1]],
        threshold=threshold
    )[0]
    
    # Find TV
    segments_info = results['segments_info']
    panoptic_seg = results['segmentation'].cpu().numpy()
    
    tv_found = []
    
    for segment in segments_info:
        label_id = segment['label_id']
        label = model.config.id2label[label_id]
        score = segment['score']
        
        if 'tv' in label.lower():
            segment_id = segment['id']
            mask = (panoptic_seg == segment_id).astype(np.uint8)
            
            # Get bounding box
            coords = np.where(mask > 0)
            if len(coords[0]) > 0:
                y_coords, x_coords = coords
                min_y, max_y = np.min(y_coords), np.max(y_coords)
                min_x, max_x = np.min(x_coords), np.max(x_coords)
                bbox = [min_x, min_y, max_x - min_x, max_y - min_y]
                
                # Calculate properties
                area = np.sum(mask)
                area_ratio = area / (h * w)
                center_y = np.mean(y_coords)
                center_x = np.mean(x_coords)
                
                # Get average color/brightness of TV region
                tv_region = image_cv[mask > 0]
                avg_brightness = np.mean(tv_region) if len(tv_region) > 0 else 0
                avg_color = np.mean(tv_region, axis=0) if len(tv_region) > 0 else [0, 0, 0]
                
                tv_found.append({
                    'mask': mask,
                    'label': label,
                    'score': score,
                    'bbox': bbox,
                    'area': area,
                    'area_ratio': area_ratio,
                    'center_y': center_y / h,
                    'center_x': center_x / w,
                    'brightness': avg_brightness,
                    'avg_color': avg_color
                })
    
    if len(tv_found) == 0:
        click.echo("\n[WARNING] No TV detected!")
        return
    
    click.echo(f"\n[OK] Found {len(tv_found)} TV(s)")
    
    # Display results
    click.echo("\n" + "="*60)
    click.echo("TV Detection Results:")
    click.echo("="*60)
    
    for idx, tv_info in enumerate(tv_found):
        click.echo(f"\nTV #{idx+1}:")
        click.echo(f"  Label          : {tv_info['label']}")
        click.echo(f"  Confidence     : {tv_info['score']:.3f} ({tv_info['score']*100:.1f}%)")
        click.echo(f"  Area           : {tv_info['area']} pixels ({tv_info['area_ratio']*100:.2f}% of image)")
        click.echo(f"  Position       : center_y={tv_info['center_y']:.2f}, center_x={tv_info['center_x']:.2f}")
        click.echo(f"  Bounding Box   : {tv_info['bbox']}")
        click.echo(f"  Brightness     : {tv_info['brightness']:.1f}")
        click.echo(f"  Avg Color (BGR): {tv_info['avg_color']}")
    
    # Visualization
    click.echo("\nCreating visualization...")
    output_image = image_cv.copy()
    
    # Color for TV
    tv_color = (0, 255, 255)  # Yellow
    
    for idx, tv_info in enumerate(tv_found):
        mask = tv_info['mask']
        bbox = tv_info['bbox']
        
        # Apply colored mask
        mask_colored = np.zeros_like(image_cv)
        mask_colored[mask > 0] = tv_color
        output_image = cv2.addWeighted(output_image, 1, mask_colored, 0.6, 0)
        
        # Draw bounding box
        x, y, bw, bh = map(int, bbox)
        cv2.rectangle(output_image, (x, y), (x+bw, y+bh), tv_color, 3)
        
        # Add label with background
        label = f"TV #{idx+1} ({tv_info['score']:.2f})"
        
        # Calculate text size for background
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        
        # Draw background rectangle
        label_y = y - 15 if y > 30 else y + bh + 30
        cv2.rectangle(output_image, 
                     (x, label_y - text_h - 10), 
                     (x + text_w + 10, label_y + baseline),
                     tv_color, -1)
        
        # Draw text
        cv2.putText(output_image, label, (x + 5, label_y - 5),
                   font, font_scale, (0, 0, 0), thickness)
        
        # Draw crosshair at center
        center_x_px = int(tv_info['center_x'] * w)
        center_y_px = int(tv_info['center_y'] * h)
        cv2.drawMarker(output_image, (center_x_px, center_y_px), 
                      tv_color, cv2.MARKER_CROSS, 20, 2)
    
    # Add title
    title = f"TV Detected: {len(tv_found)} device(s)"
    cv2.putText(output_image, title, (10, 40),
               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    
    # Save output
    cv2.imwrite(output, output_image)
    click.echo(f"\n[OK] Output saved to: {output}")
    
    # Display result
    if not no_display:
        click.echo("\nDisplaying result...")
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(1, 2, figsize=(14, 7))
            
            axes[0].imshow(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
            axes[0].set_title('Original Image')
            axes[0].axis('off')
            
            axes[1].imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
            axes[1].set_title(f'TV Detection ({len(tv_found)} found)')
            axes[1].axis('off')
            
            plt.tight_layout()
            plt.show()
        except Exception as e:
            click.echo(f"Could not display image: {str(e)}")
    
    click.echo("\n[DONE]")


if __name__ == '__main__':
    main()
