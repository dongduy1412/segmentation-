"""
Mask2Former Panoptic Segmentation
State-of-the-art model for segmenting walls, floors, ceilings, and objects
"""

import click
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import cv2
from transformers import Mask2FormerImageProcessor, Mask2FormerForUniversalSegmentation


# COCO panoptic classes that are relevant for interior spaces
INTERIOR_CLASSES = {
    'wall-brick': 'wall',
    'wall-concrete': 'wall',
    'wall-other': 'wall',
    'wall-panel': 'wall',
    'wall-stone': 'wall',
    'wall-tile': 'wall',
    'wall-wood': 'wall',
    'floor-marble': 'floor',
    'floor-other': 'floor',
    'floor-stone': 'floor',
    'floor-tile': 'floor',
    'floor-wood': 'floor',
    'ceiling-other': 'ceiling',
    'ceiling-tile': 'ceiling',
    'rug': 'floor',
    'carpet': 'floor',
    'door': 'object',
    'window': 'object',
    'couch': 'object',
    'chair': 'object',
    'table': 'object',
    'furniture': 'object',
}


def get_color_for_label(label):
    """Get consistent color for each label type"""
    colors = {
        'wall': (70, 130, 180),      # Steel blue
        'floor': (60, 179, 113),     # Medium sea green
        'ceiling': (255, 182, 193),  # Light pink
        'object': (255, 165, 0),     # Orange
        'background': (200, 200, 200), # Light gray
    }
    
    # Check if it's a specific wall/floor/ceiling type
    for key in INTERIOR_CLASSES:
        if key in label.lower():
            return colors[INTERIOR_CLASSES[key]]
    
    # Check general category
    if 'wall' in label.lower():
        return colors['wall']
    elif 'floor' in label.lower():
        return colors['floor']
    elif 'ceiling' in label.lower():
        return colors['ceiling']
    elif label.lower() in ['background', 'other', 'unlabeled']:
        return colors['background']
    else:
        return colors['object']


@click.command()
@click.option('--image_path', required=True, type=click.Path(exists=True), help='Path to input image')
@click.option('--model', default='facebook/mask2former-swin-large-coco-panoptic', 
              type=click.Choice([
                  'facebook/mask2former-swin-tiny-coco-panoptic',
                  'facebook/mask2former-swin-small-coco-panoptic',
                  'facebook/mask2former-swin-base-coco-panoptic',
                  'facebook/mask2former-swin-large-coco-panoptic',
              ]),
              help='Mask2Former model size')
@click.option('--output', default='output_mask2former.jpg', help='Output image path')
@click.option('--threshold', default=0.5, type=float, help='Confidence threshold')
@click.option('--no_display', is_flag=True, help='Skip displaying the result')
def main(image_path, model, output, threshold, no_display):
    """Segment interior spaces using Mask2Former panoptic segmentation"""
    
    click.echo("="*60)
    click.echo("Mask2Former Panoptic Segmentation")
    click.echo("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    click.echo(f"Using device: {device}")
    
    # Load image
    click.echo(f"\nLoading image: {image_path}")
    image = Image.open(image_path).convert('RGB')
    image_np = np.array(image)
    
    # Load Mask2Former model
    click.echo(f"\nLoading Mask2Former model: {model}")
    click.echo("This may take a while on first run (downloading ~1GB)...")
    
    try:
        processor = Mask2FormerImageProcessor.from_pretrained(model)
        model_m2f = Mask2FormerForUniversalSegmentation.from_pretrained(model)
        model_m2f = model_m2f.to(device)
        model_m2f.eval()
        click.echo("[OK] Model loaded successfully")
    except Exception as e:
        click.echo(f"Error loading model: {str(e)}")
        return
    
    # Process image
    click.echo("\nProcessing image...")
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Inference
    click.echo("Running panoptic segmentation...")
    with torch.no_grad():
        outputs = model_m2f(**inputs)
    
    # Post-process
    result = processor.post_process_panoptic_segmentation(
        outputs, 
        target_sizes=[image.size[::-1]]
    )[0]
    
    # Extract segmentation and segments info
    panoptic_seg = result["segmentation"].cpu().numpy()
    segments_info = result["segments_info"]
    
    click.echo(f"[OK] Found {len(segments_info)} segments")
    
    # Analyze segments
    click.echo(f"\n{'='*60}")
    click.echo("Detected Components:")
    click.echo(f"{'='*60}")
    
    category_counts = {}
    for segment in segments_info:
        label_id = segment['label_id']
        label = model_m2f.config.id2label[label_id]
        score = segment.get('score', 1.0)
        
        if score < threshold:
            continue
        
        # Categorize
        category = 'object'
        if 'wall' in label.lower():
            category = 'wall'
        elif 'floor' in label.lower() or 'rug' in label.lower() or 'carpet' in label.lower():
            category = 'floor'
        elif 'ceiling' in label.lower():
            category = 'ceiling'
        
        category_counts[category] = category_counts.get(category, 0) + 1
        
        click.echo(f"  {label:30s} (score: {score:.3f}) -> {category}")
    
    click.echo(f"\n{'='*40}")
    click.echo("Summary:")
    for cat, count in sorted(category_counts.items()):
        click.echo(f"  {cat:15s}: {count:3d} regions")
    click.echo(f"{'='*40}")
    
    # Create visualization
    click.echo("\nCreating visualization...")
    
    # Create colored segmentation map
    h, w = panoptic_seg.shape
    colored_mask = np.zeros((h, w, 3), dtype=np.uint8)
    
    for segment in segments_info:
        segment_id = segment['id']
        label_id = segment['label_id']
        label = model_m2f.config.id2label[label_id]
        score = segment.get('score', 1.0)
        
        if score < threshold:
            continue
        
        # Get mask for this segment
        mask = panoptic_seg == segment_id
        
        # Get color
        color = get_color_for_label(label)
        colored_mask[mask] = color
    
    # Blend with original image
    output_image = cv2.addWeighted(
        cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR), 
        0.4, 
        colored_mask, 
        0.6, 
        0
    )
    
    # Add legend
    legend_height = 150
    legend = np.ones((legend_height, w, 3), dtype=np.uint8) * 255
    
    y_pos = 30
    legend_items = {
        'wall': (70, 130, 180),
        'floor': (60, 179, 113),
        'ceiling': (255, 182, 193),
        'object': (255, 165, 0),
    }
    
    for label, color in legend_items.items():
        if label in category_counts:
            # Draw color box
            cv2.rectangle(legend, (20, y_pos-15), (50, y_pos), color, -1)
            cv2.rectangle(legend, (20, y_pos-15), (50, y_pos), (0, 0, 0), 1)
            
            # Draw text
            text = f"{label}: {category_counts[label]} regions"
            cv2.putText(legend, text, (60, y_pos-2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            y_pos += 30
    
    # Add model info
    model_text = f"Model: {model.split('/')[-1]}"
    cv2.putText(legend, model_text, (20, legend_height-10), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
    
    # Combine image and legend
    final_output = np.vstack([output_image, legend])
    
    # Save output
    cv2.imwrite(output, final_output)
    click.echo(f"\n[OK] Output saved to: {output}")
    
    # Display result
    if not no_display:
        click.echo("\nDisplaying result...")
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(1, 2, figsize=(14, 7))
            
            axes[0].imshow(image)
            axes[0].set_title('Original Image')
            axes[0].axis('off')
            
            axes[1].imshow(cv2.cvtColor(final_output, cv2.COLOR_BGR2RGB))
            axes[1].set_title(f'Mask2Former ({len(segments_info)} segments)')
            axes[1].axis('off')
            
            plt.tight_layout()
            plt.show()
        except Exception as e:
            click.echo(f"Could not display image: {str(e)}")
    
    click.echo("\n[DONE]")


if __name__ == '__main__':
    main()
