"""
YOLO + SAM 2 Segmentation Pipeline
Combines YOLO for object detection with SAM 2 for precise mask generation
"""

import torch
import numpy as np
from ultralytics import YOLO
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from PIL import Image
import cv2
from pathlib import Path
from typing import List, Dict, Tuple, Optional


class YOLOSAMSegmenter:
    """
    Image segmentation using YOLO for object detection and SAM 2 for mask generation
    """
    
    def __init__(
        self,
        yolo_model: str = "yolov8n.pt",
        sam_checkpoint: str = "sam2.1_hiera_small.pt",
        sam_config: str = "configs/sam2.1/sam2.1_hiera_s.yaml",
        device: str = "auto"
    ):
        """
        Initialize YOLO + SAM 2 segmentation pipeline
        
        Args:
            yolo_model: YOLO model name/path (e.g., 'yolov8n.pt', 'yolov8s.pt')
            sam_checkpoint: SAM 2 checkpoint path
            sam_config: SAM 2 config file
            device: Device to use ('cuda', 'cpu', or 'auto')
        """
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Using device: {self.device}")
        
        print(f"Loading YOLO model: {yolo_model}")
        self.yolo = YOLO(yolo_model)
        
        print("Loading SAM 2 model...")
        try:
            self.sam2_predictor = SAM2ImagePredictor.from_pretrained(
                f"facebook/sam2-hiera-small",
                device=self.device
            )
            print("SAM 2 model loaded successfully!")
        except Exception as e:
            print(f"Error loading SAM 2: {e}")
            print("Will use YOLO masks as fallback")
            self.sam2_predictor = None
            
        self.yolo_classes = self.yolo.names
        
    def detect_objects(
        self,
        image_path: str,
        conf_threshold: float = 0.25,
        classes: Optional[List[int]] = None
    ) -> List[Dict]:
        """
        Detect objects using YOLO
        
        Args:
            image_path: Path to input image
            conf_threshold: Confidence threshold for detections
            classes: List of class IDs to detect (None = all classes)
            
        Returns:
            List of detected objects with bounding boxes and class info
        """
        results = self.yolo.predict(
            image_path,
            conf=conf_threshold,
            classes=classes,
            verbose=False
        )
        
        detections = []
        for result in results:
            boxes = result.boxes
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].item()
                cls = int(box.cls[0].item())
                
                detections.append({
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'confidence': conf,
                    'class_id': cls,
                    'class_name': self.yolo_classes[cls]
                })
                
        return detections
    
    def segment_with_sam(
        self,
        image: np.ndarray,
        detections: List[Dict]
    ) -> List[Dict]:
        """
        Generate precise masks using SAM 2 based on YOLO detections
        
        Args:
            image: Input image as numpy array (RGB)
            detections: List of YOLO detections
            
        Returns:
            List of detections with added 'mask' field
        """
        if self.sam2_predictor is None:
            print("SAM 2 not available, using YOLO masks")
            return detections
            
        self.sam2_predictor.set_image(image)
        
        for detection in detections:
            bbox = detection['bbox']
            x1, y1, x2, y2 = bbox
            
            box = np.array([x1, y1, x2, y2])
            
            masks, scores, _ = self.sam2_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=box[None, :],
                multimask_output=False,
            )
            
            detection['mask'] = masks[0]
            detection['mask_score'] = scores[0]
            
        return detections
    
    def segment_image(
        self,
        image_path: str,
        conf_threshold: float = 0.25,
        classes: Optional[List[int]] = None,
        use_sam: bool = True
    ) -> Tuple[np.ndarray, List[Dict]]:
        """
        Complete segmentation pipeline
        
        Args:
            image_path: Path to input image
            conf_threshold: Confidence threshold for detections
            classes: List of class IDs to detect
            use_sam: Whether to use SAM 2 for refinement
            
        Returns:
            Tuple of (image_array, segmentation_results)
        """
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)
        
        print(f"\nDetecting objects in {Path(image_path).name}...")
        detections = self.detect_objects(image_path, conf_threshold, classes)
        print(f"Found {len(detections)} objects")
        
        if use_sam and self.sam2_predictor is not None and len(detections) > 0:
            print("Generating precise masks with SAM 2...")
            detections = self.segment_with_sam(image_np, detections)
            
        return image_np, detections
    
    def visualize_results(
        self,
        image: np.ndarray,
        detections: List[Dict],
        show_boxes: bool = True,
        show_masks: bool = True,
        show_labels: bool = True
    ) -> np.ndarray:
        """
        Visualize segmentation results
        
        Args:
            image: Input image
            detections: Segmentation results
            show_boxes: Whether to show bounding boxes
            show_masks: Whether to show masks
            show_labels: Whether to show labels
            
        Returns:
            Annotated image
        """
        result_img = image.copy()
        
        np.random.seed(42)
        colors = np.random.randint(0, 255, size=(len(self.yolo_classes), 3), dtype=np.uint8)
        
        for detection in detections:
            cls_id = detection['class_id']
            color = tuple(map(int, colors[cls_id]))
            
            if show_masks and 'mask' in detection:
                mask = detection['mask'].astype(bool)
                colored_mask = np.zeros_like(result_img)
                colored_mask[mask] = color
                result_img = cv2.addWeighted(result_img, 1.0, colored_mask, 0.5, 0)
            
            if show_boxes:
                x1, y1, x2, y2 = map(int, detection['bbox'])
                cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 2)
            
            if show_labels:
                x1, y1 = map(int, detection['bbox'][:2])
                label = f"{detection['class_name']}: {detection['confidence']:.2f}"
                
                (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(result_img, (x1, y1 - text_h - 10), (x1 + text_w, y1), color, -1)
                cv2.putText(
                    result_img,
                    label,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )
                
        return result_img
    
    def print_summary(self, detections: List[Dict]):
        """Print summary of detected objects"""
        print("\n" + "="*60)
        print("SEGMENTATION RESULTS")
        print("="*60)
        
        class_counts = {}
        for det in detections:
            cls_name = det['class_name']
            class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
            
        for cls_name, count in sorted(class_counts.items()):
            print(f"{cls_name:20} - {count} instance(s)")
            
        print("="*60)
        print(f"Total objects: {len(detections)}")
        print("="*60 + "\n")
