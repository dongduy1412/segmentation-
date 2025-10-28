"""Utility functions for image segmentation"""

import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional
import numpy as np


def save_image(image: np.ndarray, output_path: str):
    """
    Save image to file
    
    Args:
        image: Image array (RGB)
        output_path: Output file path
    """
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, image_bgr)
    print(f"Saved result to: {output_path}")


def display_image(image: np.ndarray, title: str = "Image"):
    """
    Display image using matplotlib
    
    Args:
        image: Image array (RGB)
        title: Window title
    """
    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def display_comparison(
    original: np.ndarray,
    segmented: np.ndarray,
    save_path: Optional[str] = None
):
    """
    Display original and segmented images side by side
    
    Args:
        original: Original image
        segmented: Segmented image
        save_path: Optional path to save the comparison
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    axes[0].imshow(original)
    axes[0].set_title("Original Image", fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(segmented)
    axes[1].set_title("Segmentation Result", fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison to: {save_path}")
    
    plt.show()


def get_output_path(input_path: str, suffix: str = "_segmented", extension: str = None) -> str:
    """
    Generate output path based on input path
    
    Args:
        input_path: Input file path
        suffix: Suffix to add to filename
        extension: New extension (None = keep original)
        
    Returns:
        Output file path
    """
    path = Path(input_path)
    if extension is None:
        extension = path.suffix
    else:
        extension = f".{extension}" if not extension.startswith('.') else extension
    
    output_path = path.parent / f"{path.stem}{suffix}{extension}"
    return str(output_path)
