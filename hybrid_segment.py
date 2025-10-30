"""
Hybrid Segmentation: YOLO + SAM 2 with SAM Everything fallback
Strategy:
1. Try YOLO detection first (fast, accurate for known objects)
2. If no detection with reasonable confidence -> fallback to SAM 2 Everything
3. Filter SAM Everything results to find table objects
"""

import click
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from ultralytics import YOLO
import cv2


def yolo_sam_detect(image, image_rgb, image_path, prompt, yolo_model, conf_threshold):
    """Try YOLO + SAM detection first"""
    click.echo(f"\n[Step 1] Trying YOLO detection (conf >= {conf_threshold})...")
    
    # Run YOLO detection
    results = yolo_model.predict(image_path, conf=conf_threshold, verbose=False)
    
    # Filter detections by prompt
    detections = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            class_id = int(box.cls[0])
            class_name = result.names[class_id].lower()
            confidence = float(box.conf[0])
            
            if prompt.lower() in class_name or class_name in prompt.lower():
                bbox = box.xyxy[0].cpu().numpy()
                detections.append({
                    'bbox': bbox,
                    'class_name': class_name,
                    'confidence': confidence
                })
                click.echo(f"  ✓ Found {class_name} (conf: {confidence:.2f})")
    
    if not detections:
        click.echo(f"  ✗ No objects matching '{prompt}' found with YOLO")
        return None, []
    
    click.echo(f"  ✓ YOLO detected {len(detections)} object(s)")
    
    # Load SAM 2 for refinement
    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        import sam2
        
        sam2_checkpoint = "sam2.1_b.pt"
        sam2_base = Path(sam2.__file__).parent
        model_cfg = str(sam2_base / "configs" / "sam2.1" / "sam2.1_hiera_b+.yaml")
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
        predictor = SAM2ImagePredictor(sam2_model)
        predictor.set_image(image_rgb)
        
        # Generate masks for each detection
        masks_data = []
        for idx, detection in enumerate(detections):
            bbox = detection['bbox']
            input_box = np.array([[bbox[0], bbox[1], bbox[2], bbox[3]]])
            
            masks, scores, _ = predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_box,
                multimask_output=True,
            )
            
            best_idx = np.argmax(scores)
            mask = masks[best_idx]
            score = scores[best_idx]
            
            masks_data.append({
                'mask': mask,
                'bbox': bbox,
                'confidence': detection['confidence'],
                'class_name': detection['class_name'],
                'score': score
            })
            click.echo(f"  ✓ SAM mask {idx+1}: score {score:.3f}")
        
        return 'yolo', masks_data
        
    except Exception as e:
        click.echo(f"  ✗ SAM 2 error: {str(e)}")
        return None, []


def sam_everything_fallback(image, image_rgb, min_area=500):
    """Fallback to SAM 2 Everything mode"""
    click.echo(f"\n[Step 2] Falling back to SAM 2 Everything mode...")
    
    try:
        from sam2.build_sam import build_sam2
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        import sam2
        
        sam2_checkpoint = "sam2.1_b.pt"
        sam2_base = Path(sam2.__file__).parent
        model_cfg = str(sam2_base / "configs" / "sam2.1" / "sam2.1_hiera_b+.yaml")
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
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
        
        click.echo("  Generating all possible masks...")
        masks = mask_generator.generate(image_rgb)
        click.echo(f"  ✓ Generated {len(masks)} masks")
        
        # Sort by area
        masks = sorted(masks, key=lambda x: x['area'], reverse=True)
        
        # Filter for potential table objects
        click.echo("  Filtering for table-like objects...")
        h, w = image.shape[:2]
        table_candidates = []
        
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
            height_span = (max_y - min_y) / h
            width_span = bbox[2] / w
            area_ratio = area / (h * w)
            
            # Heuristics for table detection:
            # - Not ceiling (top 20%)
            # - Not floor (bottom 20% + large area)
            # - Not wall (vertical span > 60%)
            # - Medium to large size (area ratio between 0.02 and 0.4)
            # - Positioned in center-bottom area
            
            is_ceiling = center_y < h * 0.2 and area_ratio > 0.1
            is_floor = center_y > h * 0.7 and area_ratio > 0.2 and width_span > 0.6
            is_wall = height_span > 0.6 and area_ratio > 0.15
            
            if is_ceiling or is_floor or is_wall:
                continue
            
            # Good table candidate
            if 0.02 < area_ratio < 0.4 and center_y > h * 0.3:
                confidence_score = mask_data.get('predicted_iou', 0.0)
                table_candidates.append({
                    'mask': mask,
                    'bbox': bbox,
                    'area': area,
                    'confidence': confidence_score,
                    'class_name': 'table (detected by SAM)',
                    'score': confidence_score,
                    'center_y': center_y / h,
                    'area_ratio': area_ratio
                })
        
        # Sort by area (largest first) and take top candidates
        table_candidates = sorted(table_candidates, key=lambda x: x['area'], reverse=True)[:5]
        
        click.echo(f"  ✓ Found {len(table_candidates)} table candidate(s)")
        for idx, candidate in enumerate(table_candidates):
            click.echo(f"    - Candidate {idx+1}: area_ratio={candidate['area_ratio']:.3f}, center_y={candidate['center_y']:.2f}")
        
        return 'sam_everything', table_candidates
        
    except Exception as e:
        click.echo(f"  ✗ SAM Everything error: {str(e)}")
        return None, []


@click.command()
@click.option('--image_path', required=True, type=click.Path(exists=True), help='Path to input image')
@click.option('--prompt', default='table', type=str, help='Object to detect (default: table)')
@click.option('--yolo_model', default='yolov8m.pt', type=str, help='YOLO model path')
@click.option('--conf', default=0.25, type=float, help='YOLO confidence threshold')
@click.option('--output', default='output_hybrid.jpg', help='Output image path')
@click.option('--no-display', is_flag=True, help='Skip displaying the result')
def main(image_path, prompt, yolo_model, conf, output, no_display):
    """Hybrid segmentation: YOLO+SAM with SAM Everything fallback"""
    
    click.echo("="*60)
    click.echo("Hybrid Segmentation: YOLO + SAM / SAM Everything")
    click.echo("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    click.echo(f"Device: {device}")
    click.echo(f"Target object: '{prompt}'")
    
    # Load image
    click.echo(f"\nLoading image: {image_path}")
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]
    
    # Load YOLO model
    click.echo(f"Loading YOLO model: {yolo_model}")
    yolo = YOLO(yolo_model)
    
    # Try YOLO first
    method, masks_data = yolo_sam_detect(image, image_rgb, image_path, prompt, yolo, conf)
    
    # Fallback to SAM Everything if YOLO failed
    if method is None or len(masks_data) == 0:
        method, masks_data = sam_everything_fallback(image, image_rgb, min_area=500)
    
    if method is None or len(masks_data) == 0:
        click.echo("\n✗ No objects detected by any method!")
        return
    
    # Create visualization
    click.echo(f"\n[Step 3] Creating visualization ({method})...")
    output_image = image.copy()
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255)
    ]
    
    for idx, mask_info in enumerate(masks_data):
        mask = mask_info['mask']
        color = colors[idx % len(colors)]
        
        # Apply colored mask
        mask_colored = np.zeros_like(image)
        mask_colored[mask > 0] = color
        output_image = cv2.addWeighted(output_image, 1, mask_colored, 0.5, 0)
        
        # Draw bounding box
        if 'bbox' in mask_info:
            if method == 'yolo':
                bbox = mask_info['bbox']
                x1, y1, x2, y2 = map(int, bbox)
            else:  # sam_everything
                bbox = mask_info['bbox']
                x1, y1, w_box, h_box = map(int, bbox)
                x2, y2 = x1 + w_box, y1 + h_box
            
            cv2.rectangle(output_image, (x1, y1), (x2, y2), color, 2)
            
            # Add label
            label = f"{mask_info['class_name']} {mask_info['confidence']:.2f}"
            cv2.putText(output_image, label, (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Add method indicator
    method_text = f"Method: {method.upper().replace('_', ' ')}"
    cv2.putText(output_image, method_text, (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    # Save output
    cv2.imwrite(output, output_image)
    click.echo(f"\n✓ Output saved to: {output}")
    click.echo(f"✓ Detection method: {method}")
    click.echo(f"✓ Objects found: {len(masks_data)}")
    
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
            axes[1].set_title(f'Segmented - {method.upper()} ({len(masks_data)} objects)')
            axes[1].axis('off')
            
            plt.tight_layout()
            plt.show()
        except Exception as e:
            click.echo(f"Could not display image: {str(e)}")
    
    click.echo("\n✅ Done!")


if __name__ == '__main__':
    main()
