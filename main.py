"""
Image Segmentation CLI
Supports YOLO + SAM 2 for object detection and segmentation
"""

import argparse
from pathlib import Path
from src.yolo_sam_segmenter import YOLOSAMSegmenter
from src.utils import save_image, display_comparison, get_output_path


def main():
    parser = argparse.ArgumentParser(
        description="Image Segmentation using YOLO + SAM 2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python main.py test.jpg
  
  # Save output
  python main.py test.jpg --output result.jpg
  
  # Higher confidence threshold
  python main.py test.jpg --conf 0.5
  
  # Detect specific classes (person=0, chair=56, couch=57)
  python main.py test.jpg --classes 0 56 57
  
  # Use different YOLO model
  python main.py test.jpg --yolo-model yolov8m.pt
  
  # Disable SAM refinement (faster, less accurate)
  python main.py test.jpg --no-sam
        """
    )
    
    parser.add_argument(
        "image",
        type=str,
        help="Path to input image"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Path to save output image (default: <input>_segmented.jpg)"
    )
    
    parser.add_argument(
        "--yolo-model",
        type=str,
        default="yolov8n.pt",
        choices=["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"],
        help="YOLO model to use (n=nano, s=small, m=medium, l=large, x=xlarge)"
    )
    
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for detections (default: 0.25)"
    )
    
    parser.add_argument(
        "--classes",
        type=int,
        nargs="+",
        default=None,
        help="Specific class IDs to detect (default: all classes)"
    )
    
    parser.add_argument(
        "--no-sam",
        action="store_true",
        help="Disable SAM 2 refinement (faster but less accurate)"
    )
    
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Don't display the result (only save)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to use (default: auto)"
    )
    
    args = parser.parse_args()
    
    if not Path(args.image).exists():
        print(f"Error: Image file not found: {args.image}")
        return
    
    print("="*60)
    print("IMAGE SEGMENTATION - YOLO + SAM 2")
    print("="*60)
    
    segmenter = YOLOSAMSegmenter(
        yolo_model=args.yolo_model,
        device=args.device
    )
    
    image, detections = segmenter.segment_image(
        args.image,
        conf_threshold=args.conf,
        classes=args.classes,
        use_sam=not args.no_sam
    )
    
    segmenter.print_summary(detections)
    
    result_img = segmenter.visualize_results(
        image,
        detections,
        show_boxes=True,
        show_masks=True,
        show_labels=True
    )
    
    if args.output is None:
        args.output = get_output_path(args.image, "_segmented")
    
    save_image(result_img, args.output)
    
    if not args.no_display:
        display_comparison(image, result_img, save_path=None)
    
    print("\nSegmentation complete!")


if __name__ == "__main__":
    main()
