"""
Floor Lighting Variation Detection
Uses Mask2Former to get floor mask, then analyzes lighting variations within the floor
"""

import click
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import cv2
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor
from sklearn.cluster import KMeans


def segment_floor_with_mask2former(image_path, threshold=0.5):
    """
    Use Mask2Former to detect floor regions
    """
    # Load model
    model_name = "facebook/mask2former-swin-large-coco-panoptic"
    processor = Mask2FormerImageProcessor.from_pretrained(model_name)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(model_name)
    
    # Load image
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    
    # Run inference
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Post-process
    results = processor.post_process_panoptic_segmentation(
        outputs,
        target_sizes=[image.size[::-1]],
        threshold=threshold
    )[0]
    
    # Extract floor masks
    floor_masks = []
    segments_info = results['segments_info']
    panoptic_seg = results['segmentation'].cpu().numpy()
    
    for segment in segments_info:
        label_id = segment['label_id']
        label = model.config.id2label[label_id]
        score = segment['score']
        
        # Check if it's a floor (floor, rug, carpet, etc.)
        if 'floor' in label.lower() or 'rug' in label.lower() or 'carpet' in label.lower():
            segment_id = segment['id']
            mask = (panoptic_seg == segment_id).astype(np.uint8)
            floor_masks.append({
                'mask': mask,
                'label': label,
                'score': score
            })
    
    return floor_masks, image


def analyze_lighting_regions(mask, image, n_clusters=3):
    """
    Analyze lighting variations within a floor mask
    Uses K-means clustering on brightness values
    """
    # Convert image to numpy
    image_np = np.array(image)
    
    # Get floor pixels
    floor_coords = np.where(mask > 0)
    if len(floor_coords[0]) == 0:
        return []
    
    # Extract brightness values for floor pixels
    floor_pixels = image_np[floor_coords]
    brightness = np.mean(floor_pixels, axis=1).reshape(-1, 1)
    
    # Cluster by brightness
    kmeans = KMeans(n_clusters=min(n_clusters, len(brightness)), random_state=42, n_init=10)
    labels = kmeans.fit_predict(brightness)
    
    # Create separate masks for each cluster
    lighting_regions = []
    
    for cluster_id in range(kmeans.n_clusters):
        # Get pixels belonging to this cluster
        cluster_pixels = labels == cluster_id
        
        # Create mask for this cluster
        cluster_mask = np.zeros_like(mask)
        cluster_coords_y = floor_coords[0][cluster_pixels]
        cluster_coords_x = floor_coords[1][cluster_pixels]
        cluster_mask[cluster_coords_y, cluster_coords_x] = 1
        
        # Calculate cluster properties
        cluster_brightness = kmeans.cluster_centers_[cluster_id][0]
        cluster_size = np.sum(cluster_mask)
        cluster_ratio = cluster_size / np.sum(mask)
        
        # Get bounding box
        if len(cluster_coords_y) > 0:
            min_y, max_y = np.min(cluster_coords_y), np.max(cluster_coords_y)
            min_x, max_x = np.min(cluster_coords_x), np.max(cluster_coords_x)
            bbox = [min_x, min_y, max_x - min_x, max_y - min_y]
            
            # Calculate center position
            center_y = np.mean(cluster_coords_y)
            center_x = np.mean(cluster_coords_x)
        else:
            bbox = [0, 0, 0, 0]
            center_y = 0
            center_x = 0
        
        lighting_regions.append({
            'mask': cluster_mask,
            'brightness': cluster_brightness,
            'size': cluster_size,
            'ratio': cluster_ratio,
            'bbox': bbox,
            'center_y': center_y,
            'center_x': center_x,
            'cluster_id': cluster_id
        })
    
    # Sort by brightness (darkest to brightest)
    lighting_regions = sorted(lighting_regions, key=lambda x: x['brightness'])
    
    # Label as dark, normal, bright based on position
    if len(lighting_regions) >= 3:
        lighting_regions[0]['lighting'] = 'dark'
        lighting_regions[1]['lighting'] = 'normal'
        lighting_regions[2]['lighting'] = 'bright'
    elif len(lighting_regions) == 2:
        lighting_regions[0]['lighting'] = 'dark'
        lighting_regions[1]['lighting'] = 'bright'
    elif len(lighting_regions) == 1:
        lighting_regions[0]['lighting'] = 'normal'
    
    # Add position labels (front/center/back based on y-coordinate)
    image_height = image_np.shape[0]
    for region in lighting_regions:
        center_y_ratio = region['center_y'] / image_height
        if center_y_ratio < 0.33:
            region['position'] = 'back'
        elif center_y_ratio < 0.67:
            region['position'] = 'center'
        else:
            region['position'] = 'front'
    
    return lighting_regions


@click.command()
@click.option('--image_path', required=True, type=click.Path(exists=True), help='Path to input image')
@click.option('--output', default='output_floor_lighting.jpg', help='Output image path')
@click.option('--n_clusters', default=3, type=int, help='Number of lighting zones to detect')
@click.option('--threshold', default=0.5, type=float, help='Mask2Former confidence threshold')
@click.option('--no_display', is_flag=True, help='Skip displaying the result')
def main(image_path, output, n_clusters, threshold, no_display):
    """Floor lighting variation detection"""
    
    click.echo("="*60)
    click.echo("Floor Lighting Variation Detection")
    click.echo("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    click.echo(f"Using device: {device}")
    
    # Step 1: Get floor mask using Mask2Former
    click.echo(f"\nStep 1: Detecting floors with Mask2Former...")
    floor_masks, image_pil = segment_floor_with_mask2former(image_path, threshold)
    
    if len(floor_masks) == 0:
        click.echo("[WARNING] No floor detected by Mask2Former!")
        return
    
    click.echo(f"[OK] Found {len(floor_masks)} floor region(s)")
    for idx, floor_info in enumerate(floor_masks):
        click.echo(f"  Floor {idx+1}: {floor_info['label']} (score: {floor_info['score']:.3f})")
    
    # Load original image for visualization
    image_cv = cv2.imread(image_path)
    output_image = image_cv.copy()
    
    # Step 2: Analyze lighting in each floor
    all_lighting_regions = []
    
    for floor_idx, floor_info in enumerate(floor_masks):
        floor_mask = floor_info['mask']
        floor_label = floor_info['label']
        
        click.echo(f"\nStep 2: Analyzing lighting variations in {floor_label}...")
        lighting_regions = analyze_lighting_regions(floor_mask, image_pil, n_clusters)
        
        if len(lighting_regions) == 0:
            click.echo("[WARNING] No lighting variations detected")
            continue
        
        click.echo(f"[OK] Found {len(lighting_regions)} lighting zone(s)")
        
        # Display lighting region info
        click.echo("\n" + "="*60)
        click.echo(f"Lighting Zones in {floor_label}:")
        click.echo("="*60)
        
        for idx, region in enumerate(lighting_regions):
            click.echo(f"\nZone {idx+1} ({region['lighting']} / {region['position']}):")
            click.echo(f"  Brightness    : {region['brightness']:.1f}")
            click.echo(f"  Size          : {region['size']} pixels ({region['ratio']*100:.1f}% of floor)")
            click.echo(f"  Position      : y={region['center_y']:.0f}, x={region['center_x']:.0f}")
            click.echo(f"  Bounding box  : {region['bbox']}")
        
        # Visualize
        color_map = {
            'bright': (100, 255, 100),   # Light Green
            'normal': (0, 200, 0),       # Green
            'dark': (0, 100, 0),         # Dark Green
        }
        
        for idx, region in enumerate(lighting_regions):
            lighting = region['lighting']
            color = color_map.get(lighting, (0, 200, 0))
            
            # Apply colored mask
            mask_colored = np.zeros_like(image_cv)
            mask_colored[region['mask'] > 0] = color
            output_image = cv2.addWeighted(output_image, 1, mask_colored, 0.5, 0)
            
            # Draw bounding box
            bbox = region['bbox']
            x, y, w, h = map(int, bbox)
            if w > 0 and h > 0:
                cv2.rectangle(output_image, (x, y), (x+w, y+h), color, 2)
                
                # Add label
                label = f"Floor-{lighting}-{idx+1} ({region['position']})"
                label_y = y - 10 if y > 20 else y + h + 20
                cv2.putText(output_image, label, (x, label_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        all_lighting_regions.extend(lighting_regions)
    
    # Add legend
    legend_y = 30
    cv2.putText(output_image, f"Total Floor Lighting Zones: {len(all_lighting_regions)}", 
               (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    legend_y += 30
    cv2.putText(output_image, "Light Green=Bright, Green=Normal, Dark Green=Dark", 
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
            
            axes[0].imshow(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
            axes[0].set_title('Original Image')
            axes[0].axis('off')
            
            axes[1].imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
            axes[1].set_title(f'Floor Lighting Zones ({len(all_lighting_regions)})')
            axes[1].axis('off')
            
            plt.tight_layout()
            plt.show()
        except Exception as e:
            click.echo(f"Could not display image: {str(e)}")
    
    # Summary
    click.echo("\n" + "="*60)
    click.echo("Summary:")
    click.echo("="*60)
    bright_count = sum(1 for r in all_lighting_regions if r['lighting'] == 'bright')
    normal_count = sum(1 for r in all_lighting_regions if r['lighting'] == 'normal')
    dark_count = sum(1 for r in all_lighting_regions if r['lighting'] == 'dark')
    
    front_count = sum(1 for r in all_lighting_regions if r['position'] == 'front')
    center_count = sum(1 for r in all_lighting_regions if r['position'] == 'center')
    back_count = sum(1 for r in all_lighting_regions if r['position'] == 'back')
    
    click.echo(f"  Total zones   : {len(all_lighting_regions)}")
    click.echo(f"  By lighting:")
    click.echo(f"    Bright zones: {bright_count}")
    click.echo(f"    Normal zones: {normal_count}")
    click.echo(f"    Dark zones  : {dark_count}")
    click.echo(f"  By position:")
    click.echo(f"    Front floor : {front_count}")
    click.echo(f"    Center floor: {center_count}")
    click.echo(f"    Back floor  : {back_count}")
    click.echo("\n[DONE]")


if __name__ == '__main__':
    main()
