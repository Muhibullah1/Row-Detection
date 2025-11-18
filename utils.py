"""Crop Row Detection Pipeline:
Detects and extends crop rows in agricultural field images. """

import os
import sys
import cv2
import math
import torch
import numpy as np
from predict import *
from visualize import visualize
from typing import List, Tuple, Optional
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


# Type aliases for clarity
Line = Tuple[int, int, int, int]  # (x1, y1, x2, y2)
Point = Tuple[int, int]

class CropConfig:
    """Configuration of parameters."""
    
    config = {
        'hough_threshold': 42,
        'min_line_length': 42,
        'min_area': 150,
        'name': 'canola'}
    
    @classmethod
    def get_config(cls, crop_type: str) -> dict:
        """Get configuration for specified crop type."""
        configs = {'crop': cls.config}
        return configs.get(crop_type.lower())


def calculate_line_angle(x1: int, x2: int, y1: int, y2: int) -> float:
    """
    Calculate angle of line in degrees.
    Args:
        x1, x2, y1, y2: Line coordinates
    Returns:
        Angle in degrees"""
  
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    
    # Avoid division by zero
    if dx == 0:
        dx = 1
        
    slope = dy / dx
    angle_rad = math.atan(slope)
    return math.degrees(angle_rad)


def average_angle(lines: List[Line]) -> float:
    """
    Calculate average angle of all lines.
    Args:
        lines: List of line coordinates    
    Returns:
        Average angle in degrees """
  
    if not lines:
        return 0.0
    
    angles = []
    for line in lines:
        x1, y1, x2, y2 = [max(0, int(x)) for x in line]
        
        # Ensure consistent direction (left to right)
        if x2 < x1:
            x1, y1, x2, y2 = x2, y2, x1, y1
            
        angle = calculate_line_angle(x1, x2, y1, y2)
        angles.append(angle)
    
    return sum(angles) / len(angles)


def remove_small_objects(binary_mask: np.ndarray, min_area: int) -> np.ndarray:
    """
    Remove small connected components from binary mask.
    Args:
        binary_mask: Binary image
        min_area: Minimum area threshold  
    Returns:
        Cleaned binary mask"""
  
    # Apply morphological opening
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opened = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    
    # Find removed pixels
    removed = cv2.subtract(binary_mask, opened)
    result = binary_mask.copy()
    
    # Analyze connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        removed, connectivity=8)
    
    # Remove small components
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area < min_area:
            result[labels == label] = 0
            
    return result


def aggregate_lines(lines: np.ndarray) -> List[np.ndarray]:
    """
    Cluster nearby lines and merge them.
    Args:
        lines: Array of line coordinates    
    Returns:
        List of merged line coordinates """
  
    total_lines = len(lines)
    
    # Adaptive distance threshold based on line density
    if total_lines < 10:
        distance_threshold = 50
    elif total_lines < 20:
        distance_threshold = 30
    elif total_lines < 30:
        distance_threshold = 10
    else:
        distance_threshold = 7
    
    clusters = []
    processed = set()
    
    for i, line in enumerate(lines):
        if i in processed:
            continue
            
        x1, y1, x2, y2 = line
        
        # Calculate line equation: ax + by + c = 0
        slope = (y2 - y1) / (x2 - x1 + 1e-6)
        intercept = ((y2 + y1) - slope * (x2 + x1)) / 2
        
        a = -slope
        b = 1
        c = -intercept
        d = math.sqrt(a**2 + b**2)
        
        cluster = [line]
        processed.add(i)
        
        # Find nearby lines
        for j, other_line in enumerate(lines[i+1:], start=i+1):
            if j in processed:
                continue
                
            x, y, xo, yo = other_line
            mid_x = (x + xo) / 2
            mid_y = (y + yo) / 2
            
            # Calculate perpendicular distance to line
            distance = abs(a * mid_x + b * mid_y + c) / d
            
            if distance < distance_threshold:
                cluster.append(other_line)
                processed.add(j)
        
        clusters.append(np.array(cluster))
    
    # Merge lines in each cluster by averaging
    merged_lines = [np.mean(cluster, axis=0) for cluster in clusters]
    return merged_lines


def extend_line_to_borders(image_shape: Tuple[int, ...], p1: Point, p2: Point) -> Tuple[Point, Point]:
    """
    Extend line segment to image borders.
    Args:
        image_shape: Shape of image (H, W) or (H, W, C)
        p1, p2: Line endpoints    
    Returns:
        Extended line endpoints """
      
    # Extract dimensions
    if len(image_shape) == 3:
        h, w, _ = image_shape
    else:
        h, w = image_shape
    
    x1, y1 = p1
    x2, y2 = p2
    
    # Calculate line direction and length
    dx = x2 - x1
    dy = y2 - y1
    length = math.sqrt(dx**2 + dy**2)
    
    if length == 0:
        return p1, p2
    
    # Use longer dimension for extension distance
    extension_distance = max(w, h)
    
    # Extend in both directions
    scale = extension_distance / length
    
    p3_x = int(x1 - dx * scale)
    p3_y = int(y1 - dy * scale)
    p4_x = int(x2 + dx * scale)
    p4_y = int(y2 + dy * scale)
    
    # Clip to image boundaries using line-rectangle intersection
    # For simplicity, clip coordinates directly
    p3_x = max(0, min(w, p3_x))
    p3_y = max(0, min(h, p3_y))
    p4_x = max(0, min(w, p4_x))
    p4_y = max(0, min(h, p4_y))
    
    return (p3_x, p3_y), (p4_x, p4_y)


def detect_crop_rows(image: np.ndarray, mask: np.ndarray, crop_type: str, output_path: str, output_name: str, scale_x: float = 1.0, scale_y: float = 1.0) -> Tuple[np.ndarray, Optional[List[Line]]]:
    """
    Main pipeline: detect and draw crop rows on image.
    Args:
        image: Input RGB image
        mask: Binary segmentation mask
        crop_type: Type of crop ('corn' or 'canola')
        output_path: Directory to save results
        output_name: Output filename
        scale_x, scale_y: Scale factors for output coordinates  
    Returns:
        Annotated image and list of detected lines """
      
    # Get crop-specific configuration
    config = CropConfig.get_config(crop_type)
    h, w = mask.shape
    # Preprocess mask
    mask = (mask * 255).astype(np.uint8)
    mask = remove_small_objects(mask, config['min_area'])
    mask = cv2.GaussianBlur(mask, (5, 5), 1)
    mask = cv2.Canny(mask, 100, 255)
    
    # Detect lines using Hough transform
    lines = cv2.HoughLinesP(
        mask,
        rho=1,
        theta=np.pi / 180,
        threshold=config['hough_threshold'],
        minLineLength=config['min_line_length'],
        maxLineGap=250)
    
    if lines is None:
        return image, None
    
    # Aggregate nearby lines
    lines = np.squeeze(lines, axis=1)
    merged_lines = aggregate_lines(lines)
    
    # Check if lines need rotation correction
    avg_angle = average_angle(merged_lines)
    
    if avg_angle >= 40:
        # Crop borders and re-detect
        top, bottom, left, right = 90, 90, 2, 2
        mask_cropped = mask[top:-bottom, left:-right]
        
        lines = cv2.HoughLinesP(
            mask_cropped,
            rho=1,
            theta=np.pi / 180,
            threshold=config['hough_threshold'],
            minLineLength=config['min_line_length'],
            maxLineGap=250)
        
        if lines is not None:
            lines = np.squeeze(lines, axis=1)
            merged_lines = aggregate_lines(lines)
            
            # Adjust coordinates back to original image space
            adjusted_lines = []
            for line in merged_lines:
                x1, y1, x2, y2 = line
                adjusted_lines.append([
                    x1 + left, y1 + top,
                    x2 + left, y2 + top])
            merged_lines = adjusted_lines
    
    # Extend lines to image borders and draw
    final_lines = []
    for line in merged_lines:
        x1, y1, x2, y2 = [int(v) for v in line]
        
        p1, p2 = extend_line_to_borders(image.shape, (x1, y1), (x2, y2))
        
        # Draw on image
        cv2.line(image, p1, p2, (255, 0, 0), 2)
        
        # Scale coordinates for output
        scaled_line = (
            int(p1[0] * scale_x), int(p1[1] * scale_y),
            int(p2[0] * scale_x), int(p2[1] * scale_y))
        final_lines.append(scaled_line)
    
    # Sort lines by position
    final_lines.sort(key=lambda x: (x[0], x[1]))
    
    # Save line coordinates
    output_file = os.path.join(output_path, output_name.replace('.jpg', '.txt'))
    with open(output_file, 'w') as f:
        f.write(str(final_lines))
    
    return image, final_lines


# Example 
if __name__ == "__main__":
    # Load image and mask
    image = cv2.imread('field_image.jpg')
    mask = cv2.imread('segmentation_mask.png', cv2.IMREAD_GRAYSCALE) / 255.0
    
    # Detect crop rows
    result_image, lines = detect_crop_rows(
        image=image,
        mask=mask,
        crop_type='crop_type',
        output_path='./output',
        output_name='result.jpg' )
    
    # Save result
    cv2.imwrite('result_with_detected_rows.jpg', result_image)
