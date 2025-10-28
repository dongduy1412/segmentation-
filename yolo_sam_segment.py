"""
YOLO + SAM 2 Image Segmentation
Detects objects using YOLO and refines masks using SAM 2
"""

import click
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from ultralytics import YOLO
import cv2


@click.command()
@click.option('--image_path', required=True, type=click.Path(exists=True), help='Path to input image')
@click.option('--prompt', required=True, type=str, help='Text prompt for object detection (e.g., "chair")')
@click.option('--conf', default=0.25, type=float, help='Confidence threshold for YOLO detection')
@click.option('--model', default='yolov8n.pt', type=str, help='YOLO model path')
@click.option('--no-display', is_flag=True, help='Skip displaying the result')
def main(image_path, prompt, conf, model, no_display):
    """Segment objects in image using YOLO detection and SAM 2 refinement"""
    
    click.echo("="*60)
    click.echo("YOLO + SAM 2 Image Segmentation")
    click.echo("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    click.echo(f"Using device: {device}")
    
    # Load image
    click.echo(f"\nLoading image: {image_path}")
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Load YOLO model
    click.echo(f"Loading YOLO model: {model}")
    yolo_model = YOLO(model)
    
    # Run YOLO detection
    click.echo(f"\nDetecting objects matching prompt: '{prompt}'")
    results = yolo_model.predict(image_path, conf=conf, verbose=False)
    
    # Filter detections by prompt
    detections = []
    for result in results:
        boxes = result.boxes
        for i, box in enumerate(boxes):
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
                click.echo(f"  - Found {class_name} (conf: {confidence:.2f})")
    
    if not detections:
        click.echo(f"\nNo objects matching '{prompt}' found!")
        return
    
    click.echo(f"\nTotal detections: {len(detections)}")
    
    # Load SAM 2 model
    click.echo("\nLoading SAM 2 model...")
    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        import sam2
        
        sam2_checkpoint = "sam2.1_b.pt"
        sam2_base = Path(sam2.__file__).parent
        model_cfg = str(sam2_base / "configs" / "sam2.1" / "sam2.1_hiera_b+.yaml")
        
        if not Path(sam2_checkpoint).exists():
            click.echo(f"Warning: SAM 2 checkpoint not found at {sam2_checkpoint}")
            click.echo("Falling back to YOLO masks only")
            use_sam = False
        else:
            sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
            predictor = SAM2ImagePredictor(sam2_model)
            predictor.set_image(image_rgb)
            use_sam = True
    except Exception as e:
        click.echo(f"Warning: Could not load SAM 2: {str(e)}")
        click.echo("Falling back to YOLO masks only")
        use_sam = False
    
    # Create output image
    output_image = image.copy()
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255)
    ]
    
    # Process each detection
    for idx, detection in enumerate(detections):
        bbox = detection['bbox']
        color = colors[idx % len(colors)]
        
        if use_sam:
            # Use SAM 2 to generate refined mask
            click.echo(f"Generating SAM 2 mask for detection {idx+1}...")
            input_box = np.array([[bbox[0], bbox[1], bbox[2], bbox[3]]])
            
            masks, scores, _ = predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_box,
                multimask_output=False,
            )
            
            mask = masks[0]
        else:
            # Create simple mask from bounding box
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
            x1, y1, x2, y2 = map(int, bbox)
            mask[y1:y2, x1:x2] = 1
        
        # Apply colored mask
        mask_colored = np.zeros_like(image)
        mask_colored[mask > 0] = color
        output_image = cv2.addWeighted(output_image, 1, mask_colored, 0.5, 0)
        
        # Draw bounding box
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(output_image, (x1, y1), (x2, y2), color, 2)
        
        # Add label
        label = f"{detection['class_name']} {detection['confidence']:.2f}"
        cv2.putText(output_image, label, (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Save output
    output_path = "output_yolo_sam.jpg"
    cv2.imwrite(output_path, output_image)
    click.echo(f"\n✓ Output saved to: {output_path}")
    
    # Display result
    if not no_display:
        click.echo("\nDisplaying result...")
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            axes[0].set_title('Original Image')
            axes[0].axis('off')
            
            axes[1].imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
            axes[1].set_title(f'Segmented ({len(detections)} objects)')
            axes[1].axis('off')
            
            plt.tight_layout()
            plt.show()
        except Exception as e:
            click.echo(f"Could not display image: {str(e)}")
    
    click.echo("\n✅ Done!")


if __name__ == '__main__':
    main()
