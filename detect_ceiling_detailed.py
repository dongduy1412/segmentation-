"""
Detailed Ceiling Detection with Lighting Variation Analysis
Detects multiple ceiling regions based on brightness/lighting differences
"""

import click
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import cv2
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import sam2


def analyze_ceiling_regions(masks, image, image_shape):
    """
    Analyze masks to find ceiling regions with lighting variations
    """
    h, w = image_shape[:2]
    ceiling_candidates = []
    
    for idx, mask_data in enumerate(masks):
        mask = mask_data['segmentation']
        area = mask_data['area']
        bbox = mask_data['bbox']  # [x, y, w, h]
        
        # Get mask properties
        coords = np.where(mask > 0)
        if len(coords[0]) == 0:
            continue
        
        y_coords, x_coords = coords
        center_y = np.mean(y_coords)
        center_x = np.mean(x_coords)
        
        min_y, max_y = np.min(y_coords), np.max(y_coords)
        min_x, max_x = np.min(x_coords), np.max(x_coords)
        
        height_span = (max_y - min_y) / h
        width_span = (max_x - min_x) / w
        area_ratio = area / (h * w)
        
        # Ceiling heuristics:
        # - Position: top 40% of image
        # - Horizontal extent: width > 20%
        # - Not too small: area > 1% of image
        # - Check brightness (ceiling usually brighter)
        
        is_in_ceiling_zone = center_y < h * 0.4
        has_good_width = width_span > 0.2
        has_good_area = area_ratio > 0.01
        
        if is_in_ceiling_zone and has_good_width and has_good_area:
            # Calculate brightness
            mask_region = image[mask > 0]
            avg_brightness = np.mean(mask_region) if len(mask_region) > 0 else 0
            
            ceiling_candidates.append({
                'mask': mask,
                'bbox': bbox,
                'area': area,
                'area_ratio': area_ratio,
                'center_y': center_y / h,
                'center_x': center_x / w,
                'width_span': width_span,
                'height_span': height_span,
                'brightness': avg_brightness,
                'score': mask_data.get('predicted_iou', 0.0)
            })
    
    # Sort by area (largest first)
    ceiling_candidates = sorted(ceiling_candidates, key=lambda x: x['area'], reverse=True)
    
    # Group by brightness to find different lighting zones
    if len(ceiling_candidates) > 0:
        ceiling_candidates = analyze_brightness_groups(ceiling_candidates)
    
    return ceiling_candidates


def analyze_brightness_groups(candidates):
    """
    Group ceiling candidates by brightness to identify lighting variations
    """
    if len(candidates) == 0:
        return []
    
    # Calculate brightness statistics
    brightnesses = [c['brightness'] for c in candidates]
    avg_brightness = np.mean(brightnesses)
    std_brightness = np.std(brightnesses)
    
    # Label each candidate with brightness category
    for candidate in candidates:
        brightness_diff = candidate['brightness'] - avg_brightness
        
        if brightness_diff > std_brightness:
            candidate['lighting'] = 'bright'
        elif brightness_diff < -std_brightness:
            candidate['lighting'] = 'dark'
        else:
            candidate['lighting'] = 'normal'
    
    return candidates


@click.command()
@click.option('--image_path', required=True, type=click.Path(exists=True), help='Path to input image')
@click.option('--output', default='output_ceiling_detailed.jpg', help='Output image path')
@click.option('--min_area', default=100, type=int, help='Minimum mask area')
@click.option('--no_display', is_flag=True, help='Skip displaying the result')
def main(image_path, output, min_area, no_display):
    """Detailed ceiling detection with lighting variation analysis"""
    
    click.echo("="*60)
    click.echo("Detailed Ceiling Detection")
    click.echo("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    click.echo(f"Using device: {device}")
    
    # Load image
    click.echo(f"\nLoading image: {image_path}")
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]
    
    # Load SAM 2 model
    click.echo("\nLoading SAM 2 Automatic Mask Generator...")
    try:
        sam2_checkpoint = "sam2.1_b.pt"
        sam2_base = Path(sam2.__file__).parent
        model_cfg = str(sam2_base / "configs" / "sam2.1" / "sam2.1_hiera_b+.yaml")
        
        sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
        
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
        click.echo(f"[ERROR] Failed to load SAM 2: {str(e)}")
        return
    
    # Generate masks
    click.echo("\nGenerating all possible masks...")
    masks = mask_generator.generate(image_rgb)
    click.echo(f"[OK] Generated {len(masks)} masks")
    
    # Analyze ceiling regions
    click.echo("\nAnalyzing ceiling regions with lighting variations...")
    ceiling_regions = analyze_ceiling_regions(masks, image, image.shape)
    
    if len(ceiling_regions) == 0:
        click.echo("[WARNING] No ceiling regions detected!")
        return
    
    click.echo(f"[OK] Found {len(ceiling_regions)} ceiling region(s)")
    
    # Display results
    click.echo("\n" + "="*60)
    click.echo("Ceiling Regions Detected:")
    click.echo("="*60)
    
    for idx, region in enumerate(ceiling_regions):
        click.echo(f"\nRegion {idx+1}:")
        click.echo(f"  Area ratio     : {region['area_ratio']:.3f} ({region['area_ratio']*100:.1f}% of image)")
        click.echo(f"  Position       : center_y={region['center_y']:.2f}, center_x={region['center_x']:.2f}")
        click.echo(f"  Size           : width={region['width_span']:.2f}, height={region['height_span']:.2f}")
        click.echo(f"  Brightness     : {region['brightness']:.1f}")
        click.echo(f"  Lighting zone  : {region['lighting']}")
        click.echo(f"  Score          : {region['score']:.3f}")
    
    # Create visualization
    click.echo("\nCreating visualization...")
    output_image = image.copy()
    
    # Define colors for different lighting zones
    color_map = {
        'bright': (255, 255, 0),   # Yellow
        'normal': (255, 0, 255),   # Magenta
        'dark': (128, 0, 255),     # Purple
    }
    
    # Draw each ceiling region
    for idx, region in enumerate(ceiling_regions):
        mask = region['mask']
        lighting = region['lighting']
        color = color_map.get(lighting, (255, 0, 255))
        
        # Apply colored mask
        mask_colored = np.zeros_like(image)
        mask_colored[mask > 0] = color
        output_image = cv2.addWeighted(output_image, 1, mask_colored, 0.5, 0)
        
        # Draw bounding box
        bbox = region['bbox']
        x, y, bw, bh = map(int, bbox)
        cv2.rectangle(output_image, (x, y), (x+bw, y+bh), color, 2)
        
        # Add label
        label = f"Ceiling-{idx+1} ({lighting})"
        cv2.putText(output_image, label, (x, y-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Add legend
    legend_y = 30
    cv2.putText(output_image, f"Total Ceiling Regions: {len(ceiling_regions)}", 
               (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    legend_y += 30
    cv2.putText(output_image, "Yellow=Bright, Magenta=Normal, Purple=Dark", 
               (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Save output
    cv2.imwrite(output, output_image)
    click.echo(f"\n[OK] Output saved to: {output}")
    
    # Display result
    if not no_display:
        click.echo("\nDisplaying result...")
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(1, 2, figsize=(14, 7))
            
            axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            axes[0].set_title('Original Image')
            axes[0].axis('off')
            
            axes[1].imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
            axes[1].set_title(f'Ceiling Regions ({len(ceiling_regions)} zones)')
            axes[1].axis('off')
            
            plt.tight_layout()
            plt.show()
        except Exception as e:
            click.echo(f"Could not display image: {str(e)}")
    
    # Summary
    click.echo("\n" + "="*60)
    click.echo("Summary:")
    click.echo("="*60)
    bright_count = sum(1 for r in ceiling_regions if r['lighting'] == 'bright')
    normal_count = sum(1 for r in ceiling_regions if r['lighting'] == 'normal')
    dark_count = sum(1 for r in ceiling_regions if r['lighting'] == 'dark')
    
    click.echo(f"  Total regions  : {len(ceiling_regions)}")
    click.echo(f"  Bright zones   : {bright_count}")
    click.echo(f"  Normal zones   : {normal_count}")
    click.echo(f"  Dark zones     : {dark_count}")
    click.echo("\n[DONE]")


if __name__ == '__main__':
    main()
