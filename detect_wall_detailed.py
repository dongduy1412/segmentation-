"""
Detailed Wall Detection with Lighting Variation Analysis
Detects multiple wall regions based on brightness/lighting differences
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


def analyze_wall_regions(masks, image, image_shape):
    """
    Analyze masks to find wall regions with lighting variations
    """
    h, w = image_shape[:2]
    wall_candidates = []
    
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
        
        # Wall heuristics:
        # - Position: middle to upper portion (not top 10% = ceiling, not bottom 25% = floor)
        # - Vertical extent: height > 25% (walls are typically tall)
        # - Area: typically > 2% of image
        # - Not too wide relative to height (aspect ratio check)
        
        is_in_wall_zone = 0.10 < center_y < 0.75
        has_good_height = height_span > 0.20
        has_good_area = area_ratio > 0.015
        
        # Aspect ratio check - walls usually taller than wide (or very wide background)
        aspect_ratio = height_span / (width_span + 0.001)  # avoid division by zero
        is_wall_like = aspect_ratio > 0.4 or width_span > 0.4
        
        # Skip very small regions and ceiling/floor obvious candidates
        not_ceiling = center_y > h * 0.15 or width_span < 0.9
        not_floor = center_y < h * 0.8 or area_ratio < 0.15
        
        if is_in_wall_zone and has_good_height and has_good_area and is_wall_like and not_ceiling and not_floor:
            # Calculate brightness
            mask_region = image[mask > 0]
            avg_brightness = np.mean(mask_region) if len(mask_region) > 0 else 0
            
            # Determine position (left, center, right based on center_x)
            if center_x < w * 0.33:
                position = 'left'
            elif center_x > w * 0.67:
                position = 'right'
            else:
                position = 'center'
            
            wall_candidates.append({
                'mask': mask,
                'bbox': bbox,
                'area': area,
                'area_ratio': area_ratio,
                'center_y': center_y / h,
                'center_x': center_x / w,
                'width_span': width_span,
                'height_span': height_span,
                'aspect_ratio': aspect_ratio,
                'brightness': avg_brightness,
                'position': position,
                'score': mask_data.get('predicted_iou', 0.0)
            })
    
    # Sort by area (largest first)
    wall_candidates = sorted(wall_candidates, key=lambda x: x['area'], reverse=True)
    
    # Group by brightness to find different lighting zones
    if len(wall_candidates) > 0:
        wall_candidates = analyze_brightness_groups(wall_candidates)
    
    return wall_candidates


def analyze_brightness_groups(candidates):
    """
    Group wall candidates by brightness to identify lighting variations
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
        
        if brightness_diff > std_brightness * 0.5:
            candidate['lighting'] = 'bright'
        elif brightness_diff < -std_brightness * 0.5:
            candidate['lighting'] = 'dark'
        else:
            candidate['lighting'] = 'normal'
    
    return candidates


@click.command()
@click.option('--image_path', required=True, type=click.Path(exists=True), help='Path to input image')
@click.option('--output', default='output_wall_detailed.jpg', help='Output image path')
@click.option('--min_area', default=100, type=int, help='Minimum mask area')
@click.option('--no_display', is_flag=True, help='Skip displaying the result')
def main(image_path, output, min_area, no_display):
    """Detailed wall detection with lighting variation analysis"""
    
    click.echo("="*60)
    click.echo("Detailed Wall Detection")
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
    
    # Analyze wall regions
    click.echo("\nAnalyzing wall regions with lighting variations...")
    wall_regions = analyze_wall_regions(masks, image, image.shape)
    
    if len(wall_regions) == 0:
        click.echo("[WARNING] No wall regions detected!")
        return
    
    click.echo(f"[OK] Found {len(wall_regions)} wall region(s)")
    
    # Display results
    click.echo("\n" + "="*60)
    click.echo("Wall Regions Detected:")
    click.echo("="*60)
    
    for idx, region in enumerate(wall_regions):
        click.echo(f"\nRegion {idx+1}:")
        click.echo(f"  Position       : {region['position']} wall")
        click.echo(f"  Area ratio     : {region['area_ratio']:.3f} ({region['area_ratio']*100:.1f}% of image)")
        click.echo(f"  Center         : y={region['center_y']:.2f}, x={region['center_x']:.2f}")
        click.echo(f"  Size           : width={region['width_span']:.2f}, height={region['height_span']:.2f}")
        click.echo(f"  Aspect ratio   : {region['aspect_ratio']:.2f}")
        click.echo(f"  Brightness     : {region['brightness']:.1f}")
        click.echo(f"  Lighting zone  : {region['lighting']}")
        click.echo(f"  Score          : {region['score']:.3f}")
    
    # Create visualization
    click.echo("\nCreating visualization...")
    output_image = image.copy()
    
    # Define colors for different lighting zones
    color_map = {
        'bright': (100, 200, 255),   # Light Blue
        'normal': (0, 100, 255),     # Blue
        'dark': (0, 50, 150),        # Dark Blue
    }
    
    # Additional colors by position
    position_modifier = {
        'left': (0, 50, 50),
        'center': (0, 0, 0),
        'right': (50, 0, 50),
    }
    
    # Draw each wall region
    for idx, region in enumerate(wall_regions):
        mask = region['mask']
        lighting = region['lighting']
        position = region['position']
        
        # Get base color and modify slightly by position
        base_color = color_map.get(lighting, (0, 100, 255))
        color = tuple(np.clip(np.array(base_color) + np.array(position_modifier[position]), 0, 255).astype(int))
        
        # Apply colored mask
        mask_colored = np.zeros_like(image)
        mask_colored[mask > 0] = color
        output_image = cv2.addWeighted(output_image, 1, mask_colored, 0.5, 0)
        
        # Draw bounding box
        bbox = region['bbox']
        x, y, bw, bh = map(int, bbox)
        cv2.rectangle(output_image, (x, y), (x+bw, y+bh), color, 2)
        
        # Add label
        label = f"Wall-{idx+1} ({position}/{lighting})"
        label_y = y - 10 if y > 20 else y + bh + 20
        cv2.putText(output_image, label, (x, label_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Add legend
    legend_y = 30
    cv2.putText(output_image, f"Total Wall Regions: {len(wall_regions)}", 
               (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    legend_y += 30
    cv2.putText(output_image, "Light Blue=Bright, Blue=Normal, Dark Blue=Dark", 
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
            axes[1].set_title(f'Wall Regions ({len(wall_regions)} zones)')
            axes[1].axis('off')
            
            plt.tight_layout()
            plt.show()
        except Exception as e:
            click.echo(f"Could not display image: {str(e)}")
    
    # Summary
    click.echo("\n" + "="*60)
    click.echo("Summary:")
    click.echo("="*60)
    bright_count = sum(1 for r in wall_regions if r['lighting'] == 'bright')
    normal_count = sum(1 for r in wall_regions if r['lighting'] == 'normal')
    dark_count = sum(1 for r in wall_regions if r['lighting'] == 'dark')
    
    left_count = sum(1 for r in wall_regions if r['position'] == 'left')
    center_count = sum(1 for r in wall_regions if r['position'] == 'center')
    right_count = sum(1 for r in wall_regions if r['position'] == 'right')
    
    click.echo(f"  Total regions  : {len(wall_regions)}")
    click.echo(f"  By lighting:")
    click.echo(f"    Bright zones : {bright_count}")
    click.echo(f"    Normal zones : {normal_count}")
    click.echo(f"    Dark zones   : {dark_count}")
    click.echo(f"  By position:")
    click.echo(f"    Left walls   : {left_count}")
    click.echo(f"    Center walls : {center_count}")
    click.echo(f"    Right walls  : {right_count}")
    click.echo("\n[DONE]")


if __name__ == '__main__':
    main()
