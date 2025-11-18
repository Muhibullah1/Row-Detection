import cv2
import math
import torch
import numpy as np
from Utils import *
import cv_algorithms
from PIL import Image
import torch.nn as nn
import matplotlib.pyplot as plt

def predict(model, image_path, output_image_name, output_path, device='cpu'):
    """
    Predict segmentation mask and detect lines in the image. 
    Args:
        model: Trained segmentation model
        image_path: Path to input image
        output_image_name: Name for output image file
        output_path: Directory to save outputs
        device: Computing device ('cpu' or 'cuda')
    Returns:
        final_lines: List of detected line coordinates in original image scale"""
    
    sigmoid = nn.Sigmoid()  
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Store original dimensions
    original_height, original_width = image.shape[:2]
    
    model_input_size = (600, 600)
    resized_image = cv2.resize(image, model_input_size)
    
    # Calculate scale factors for converting back to original size
    height_scale = original_height / model_input_size[0]
    width_scale = original_width / model_input_size[1]
    
    # Prepare image for model
    normalized_image = resized_image / 255.0
    normalized_image = normalized_image.transpose((2, 0, 1))  # HWC to CHW
    input_tensor = torch.from_numpy(normalized_image).float()
    
    # Run inference
    with torch.no_grad():
        model.eval()
        model.to(device)
        input_batch = input_tensor.to(device).unsqueeze(0)
        predicted_mask = model(input_batch)
        predicted_mask = predicted_mask.squeeze().cpu()
        predicted_mask = sigmoid(predicted_mask).round()
        mask_array = predicted_mask.numpy()
        
        # Threshold and skeletonize
        binary_mask = cv2.threshold(mask_array, 0, 1, cv2.THRESH_BINARY)[1]
        skeleton = cv_algorithms.guo_hall(binary_mask.astype(np.uint8))
    
    # Detect and draw lines
    output_image, final_lines = draw_lines(
        np.copy(resized_image), 
        skeleton, 
        height_scale, 
        width_scale, 
        output_image_name, 
        output_path)
    
    if final_lines == 0:
        return None
    
    # Resize outputs back to original dimensions
    skeleton = cv2.resize(skeleton, (original_width, original_height))
    output_image = cv2.resize(output_image, (original_width, original_height))
    
    return final_lines


def draw_lines(image, mask, height_scale, width_scale, output_image_name, output_path):
    """
    Detect lines in skeleton mask and draw them on image.
    Args:
        image: Input image to draw lines on
        mask: Binary skeleton mask
        height_scale: Scale factor for height (original/resized)
        width_scale: Scale factor for width (original/resized)
        output_image_name: Name for output file
        output_path: Directory to save outputs
    Returns:
        image: Image with drawn lines
        final_lines: List of line coordinates scaled to original image size """
    
    # Prepare mask for line detection
    mask = mask * 255
    mask = cv2.GaussianBlur(mask, (5, 5), 1)
    edges = cv2.Canny(mask.astype(np.uint8), 100, 255)
    
    # Detect lines using Hough transform
    lines = cv2.HoughLinesP(
        edges, 
        rho=1, 
        theta=np.pi / 180, 
        threshold=50,
        minLineLength=50, 
        maxLineGap=250)
    
    if not np.any(lines):
        return 0, 0
    
    lines = np.squeeze(lines, axis=1)
    merged_lines = aggregate_lines(lines)
    final_lines = []
    
    # Draw lines and calculate properties
    for line in merged_lines:
        x1, y1, x2, y2 = line.astype(int)
        
        # Calculate slope (for future use if needed)
        slope = math.atan((y2 - y1) / (x2 - x1 + 0.0001))
        length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
        # Draw line on image
        cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
    
    # Extend lines to image boundaries and scale to original size
    image_height, image_width = image.shape[:2]
    
    for line in merged_lines:
        x1, y1, x2, y2 = line.astype(int)
        point1 = (x1, y1)
        point2 = (x2, y2)
        
        # Extend line to image boundaries
        extended_point1, extended_point2 = extend_line(image.shape, point1, point2)
        
        # Scale coordinates back to original image size
        scaled_x1 = extended_point1[0] * width_scale
        scaled_y1 = extended_point1[1] * height_scale
        scaled_x2 = extended_point2[0] * width_scale
        scaled_y2 = extended_point2[1] * height_scale
        
        # Ensure consistent ordering (top point first)
        if extended_point1[1] <= extended_point2[1]:
            final_lines.append((scaled_x1, scaled_y1, scaled_x2, scaled_y2))
        else:
            final_lines.append((scaled_x2, scaled_y2, scaled_x1, scaled_y1))
    
    # Sort lines by position (x, then y)
    final_lines = sorted(final_lines, key=lambda line: (line[0], line[1]))
    
    # Save line coordinates to file
    output_file_path = output_path + output_image_name[:-4] + ".txt"
    with open(output_file_path, "w") as file:
        file.write(str(final_lines))
    
    return image, final_lines
