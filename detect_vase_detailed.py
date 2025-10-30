"""
Detailed Vase Detection
Uses Mask2Former to detect all vases with detailed visualization
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
@click.option('--output', default='output_vase_detailed.jpg', help='Output image path')
@click.option('--threshold', default=0.5, type=float, help='Confidence threshold')
@click.option('--no_display', is_flag=True, help='Skip displaying the result')
def main(image_path, output, threshold, no_display):
    """Detailed vase detection"""
    
    click.echo("="*60)
    click.echo("Detailed Vase Detection")
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
    
    # Find vases
    segments_info = results['segments_info']
    panoptic_seg = results['segmentation'].cpu().numpy()
    
    vases_found = []
    
    for segment in segments_info:
        label_id = segment['label_id']
        label = model.config.id2label[label_id]
        score = segment['score']
        
        if 'vase' in label.lower():
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
                height = max_y - min_y
                width = max_x - min_x
                
                # Get average color/brightness
                vase_region = image_cv[mask > 0]
                avg_brightness = np.mean(vase_region) if len(vase_region) > 0 else 0
                avg_color = np.mean(vase_region, axis=0) if len(vase_region) > 0 else [0, 0, 0]
                
                # Determine position (left/center/right, top/middle/bottom)
                if center_x < w * 0.33:
                    pos_x = 'left'
                elif center_x > w * 0.67:
                    pos_x = 'right'
                else:
                    pos_x = 'center'
                
                if center_y < h * 0.33:
                    pos_y = 'top'
                elif center_y > h * 0.67:
                    pos_y = 'bottom'
                else:
                    pos_y = 'middle'
                
                vases_found.append({
                    'mask': mask,
                    'label': label,
                    'score': score,
                    'bbox': bbox,
                    'area': area,
                    'area_ratio': area_ratio,
                    'center_y': center_y / h,
                    'center_x': center_x / w,
                    'height': height,
                    'width': width,
                    'brightness': avg_brightness,
                    'avg_color': avg_color,
                    'position': f"{pos_y}-{pos_x}"
                })
    
    if len(vases_found) == 0:
        click.echo("\n[WARNING] No vases detected!")
        return
    
    # Sort by area (largest first)
    vases_found = sorted(vases_found, key=lambda x: x['area'], reverse=True)
    
    click.echo(f"\n[OK] Found {len(vases_found)} vase(s)")
    
    # Display results
    click.echo("\n" + "="*60)
    click.echo("Vase Detection Results:")
    click.echo("="*60)
    
    for idx, vase_info in enumerate(vases_found):
        click.echo(f"\nVase #{idx+1}:")
        click.echo(f"  Label          : {vase_info['label']}")
        click.echo(f"  Confidence     : {vase_info['score']:.3f} ({vase_info['score']*100:.1f}%)")
        click.echo(f"  Area           : {vase_info['area']} pixels ({vase_info['area_ratio']*100:.2f}% of image)")
        click.echo(f"  Size           : {vase_info['width']} x {vase_info['height']} pixels")
        click.echo(f"  Position       : {vase_info['position']} (y={vase_info['center_y']:.2f}, x={vase_info['center_x']:.2f})")
        click.echo(f"  Bounding Box   : {vase_info['bbox']}")
        click.echo(f"  Brightness     : {vase_info['brightness']:.1f}")
        click.echo(f"  Avg Color (BGR): [{vase_info['avg_color'][0]:.1f}, {vase_info['avg_color'][1]:.1f}, {vase_info['avg_color'][2]:.1f}]")
    
    # Visualization
    click.echo("\nCreating visualization...")
    output_image = image_cv.copy()
    
    # Different colors for each vase
    colors = [
        (255, 0, 255),    # Magenta
        (0, 255, 255),    # Yellow
        (255, 128, 0),    # Orange
        (128, 0, 255),    # Purple
        (0, 255, 128),    # Cyan-Green
        (255, 0, 128),    # Pink
    ]
    
    for idx, vase_info in enumerate(vases_found):
        mask = vase_info['mask']
        bbox = vase_info['bbox']
        color = colors[idx % len(colors)]
        
        # Apply colored mask
        mask_colored = np.zeros_like(image_cv)
        mask_colored[mask > 0] = color
        output_image = cv2.addWeighted(output_image, 1, mask_colored, 0.6, 0)
        
        # Draw bounding box
        x, y, bw, bh = map(int, bbox)
        cv2.rectangle(output_image, (x, y), (x+bw, y+bh), color, 2)
        
        # Add label with background
        label = f"Vase #{idx+1} ({vase_info['score']:.2f})"
        
        # Calculate text size for background
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        
        # Draw background rectangle
        label_y = y - 10 if y > 30 else y + bh + 25
        cv2.rectangle(output_image, 
                     (x, label_y - text_h - 8), 
                     (x + text_w + 8, label_y + baseline),
                     color, -1)
        
        # Draw text
        cv2.putText(output_image, label, (x + 4, label_y - 4),
                   font, font_scale, (0, 0, 0), thickness)
        
        # Draw center marker
        center_x_px = int(vase_info['center_x'] * w)
        center_y_px = int(vase_info['center_y'] * h)
        cv2.drawMarker(output_image, (center_x_px, center_y_px), 
                      color, cv2.MARKER_DIAMOND, 15, 2)
    
    # Add title
    title = f"Vases Detected: {len(vases_found)} item(s)"
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
            axes[1].set_title(f'Vase Detection ({len(vases_found)} found)')
            axes[1].axis('off')
            
            plt.tight_layout()
            plt.show()
        except Exception as e:
            click.echo(f"Could not display image: {str(e)}")
    
    # Summary
    click.echo("\n" + "="*60)
    click.echo("Summary:")
    click.echo("="*60)
    click.echo(f"  Total vases    : {len(vases_found)}")
    click.echo(f"  Average size   : {np.mean([v['area'] for v in vases_found]):.0f} pixels")
    click.echo(f"  Size range     : {min([v['area'] for v in vases_found])} - {max([v['area'] for v in vases_found])} pixels")
    click.echo(f"  Avg confidence : {np.mean([v['score'] for v in vases_found]):.3f}")
    
    click.echo("\n[DONE]")


if __name__ == '__main__':
    main()
