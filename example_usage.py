"""
Example usage of the image segmentation library
"""

from src.yolo_sam_segmenter import YOLOSAMSegmenter
from src.utils import save_image, display_comparison, get_output_path


def example_basic_segmentation():
    """Basic segmentation example"""
    print("="*60)
    print("Example 1: Basic Image Segmentation")
    print("="*60)
    
    segmenter = YOLOSAMSegmenter(
        yolo_model="yolov8n.pt",
        device="auto"
    )
    
    image, detections = segmenter.segment_image(
        "test.jpg",
        conf_threshold=0.25
    )
    
    segmenter.print_summary(detections)
    
    result_img = segmenter.visualize_results(image, detections)
    save_image(result_img, "example_basic.jpg")
    print()


def example_furniture_only():
    """Segment only furniture items"""
    print("="*60)
    print("Example 2: Furniture Only (chairs, couches, tables)")
    print("="*60)
    
    furniture_classes = [
        56,  # chair
        57,  # couch
        60,  # dining table
        61,  # toilet
    ]
    
    segmenter = YOLOSAMSegmenter(yolo_model="yolov8n.pt")
    
    image, detections = segmenter.segment_image(
        "test.jpg",
        conf_threshold=0.25,
        classes=furniture_classes
    )
    
    segmenter.print_summary(detections)
    
    result_img = segmenter.visualize_results(image, detections)
    save_image(result_img, "example_furniture.jpg")
    print()


def example_high_confidence():
    """Only show high-confidence detections"""
    print("="*60)
    print("Example 3: High Confidence Only (0.5+)")
    print("="*60)
    
    segmenter = YOLOSAMSegmenter(yolo_model="yolov8n.pt")
    
    image, detections = segmenter.segment_image(
        "test.jpg",
        conf_threshold=0.5
    )
    
    segmenter.print_summary(detections)
    
    result_img = segmenter.visualize_results(image, detections)
    save_image(result_img, "example_high_conf.jpg")
    print()


def example_without_sam():
    """Fast segmentation without SAM refinement"""
    print("="*60)
    print("Example 4: Fast Mode (YOLO only, no SAM)")
    print("="*60)
    
    segmenter = YOLOSAMSegmenter(yolo_model="yolov8n.pt")
    
    image, detections = segmenter.segment_image(
        "test.jpg",
        conf_threshold=0.25,
        use_sam=False
    )
    
    segmenter.print_summary(detections)
    
    result_img = segmenter.visualize_results(image, detections)
    save_image(result_img, "example_fast.jpg")
    print()


def example_custom_visualization():
    """Custom visualization options"""
    print("="*60)
    print("Example 5: Custom Visualization (masks only)")
    print("="*60)
    
    segmenter = YOLOSAMSegmenter(yolo_model="yolov8n.pt")
    
    image, detections = segmenter.segment_image("test.jpg")
    
    result_img = segmenter.visualize_results(
        image,
        detections,
        show_boxes=False,
        show_masks=True,
        show_labels=False
    )
    
    save_image(result_img, "example_masks_only.jpg")
    print()


def example_processing_detections():
    """Process detection results programmatically"""
    print("="*60)
    print("Example 6: Processing Detection Data")
    print("="*60)
    
    segmenter = YOLOSAMSegmenter(yolo_model="yolov8n.pt")
    
    image, detections = segmenter.segment_image("test.jpg")
    
    for i, det in enumerate(detections, 1):
        print(f"\nObject {i}:")
        print(f"  Class: {det['class_name']}")
        print(f"  Confidence: {det['confidence']:.2%}")
        print(f"  Bounding Box: {det['bbox']}")
        
        if 'mask' in det:
            mask = det['mask']
            area = mask.sum()
            print(f"  Mask Area: {area} pixels")
            print(f"  Mask Score: {det.get('mask_score', 'N/A')}")
    
    print()


if __name__ == "__main__":
    print("\n" + "="*60)
    print("IMAGE SEGMENTATION - USAGE EXAMPLES")
    print("="*60 + "\n")
    
    # Run all examples
    examples = [
        ("Basic Segmentation", example_basic_segmentation),
        ("Furniture Only", example_furniture_only),
        ("High Confidence", example_high_confidence),
        ("Fast Mode", example_without_sam),
        ("Custom Visualization", example_custom_visualization),
        ("Processing Detections", example_processing_detections),
    ]
    
    print("Available examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")
    
    print("\nRunning example 1 (Basic Segmentation)...")
    print("To run other examples, edit this file and call the desired function.\n")
    
    example_basic_segmentation()
    
    print("Done! Check the output images in the current directory.")
