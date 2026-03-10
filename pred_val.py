#!/usr/bin/env python3
"""
Command‑line interface for CRowNet: predict on images or validate a trained model.
"""

import os
import sys
import argparse
import torch
import cv2
import numpy as np
from torch.utils.data import DataLoader

# Import your project modules
from model import CRowNet
from dataloader import DatasetLoader
from predict import predict as predict_image   # rename to avoid conflict
# If the above import fails, ensure the module names match your files.

# Import validation functions from train.py (adjust import if needed)
try:
    from train import validate_model, compute_iou, compute_dice, compute_pixel_accuracy
except ImportError:
    # If train.py is not structured as a module, copy the needed functions here
    # For completeness, we'll include them inline below (commented out).
    pass

# ----------------------------------------------------------------------
# Helper: validation dataloader setup (similar to main.py)
# ----------------------------------------------------------------------
def get_val_transform():
    """Validation transform: only resize to 567x567."""
    import albumentations as aug
    return [aug.Resize(567, 567)]

def setup_val_dataloader(images_dir, masks_dir, batch_size=4):
    """
    Create a DataLoader for validation.
    Expects images_dir to contain a 'val' subfolder with images,
    and masks_dir to contain corresponding masks (same filenames).
    """
    val_images_dir = os.path.join(images_dir, 'val')
    if not os.path.isdir(val_images_dir):
        raise FileNotFoundError(f"Validation image directory not found: {val_images_dir}")
    if not os.path.isdir(masks_dir):
        raise FileNotFoundError(f"Masks directory not found: {masks_dir}")

    image_files = sorted(os.listdir(val_images_dir))
    # Filter to only images that exist in masks_dir
    mask_files = [f for f in image_files if os.path.isfile(os.path.join(masks_dir, f))]
    if not mask_files:
        raise RuntimeError("No matching mask files found.")

    # Create dataset
    dataset = DatasetLoader(
        mask_files,           # list of filenames
        val_images_dir,       # image directory
        masks_dir,            # mask directory
        transform=get_val_transform()
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    print(f"Validation samples: {len(dataset)}")
    return loader


# ----------------------------------------------------------------------
# Prediction subcommand
# ----------------------------------------------------------------------
def add_predict_parser(subparsers):
    parser = subparsers.add_parser('predict', help='Run inference on one or more images')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model checkpoint (.pth)')
    parser.add_argument('--input', type=str, required=True,
                        help='Input image file or directory containing images')
    parser.add_argument('--output_dir', type=str, default='./predictions',
                        help='Directory to save results (default: ./predictions)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use: "cuda" or "cpu". Auto‑detect if not set.')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for directory processing (default: 1)')
    parser.add_argument('--crop_type', type=str, default='canola',
                        help='Crop type for line detection (default: canola)')
    return parser


def run_predict(args):
    """Execute prediction on a single image or a directory."""
    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    model = CRowNet(in_channels=3, out_channels=1)
    checkpoint = torch.load(args.model, map_location=device)
    # Handle different checkpoint formats (full dict or state_dict)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    print(f"Model loaded from {args.model}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Determine input mode
    if os.path.isfile(args.input):
        # Single image
        print(f"Processing single image: {args.input}")
        final_lines = predict_image(
            model=model,
            image_path=args.input,
            output_image_name=os.path.basename(args.input),
            output_path=args.output_dir,
            device=device
        )
        if final_lines is None:
            print("No lines detected.")
        else:
            print(f"Detected {len(final_lines)} lines.")

    elif os.path.isdir(args.input):
        # Directory of images
        image_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif')
        image_files = [f for f in os.listdir(args.input)
                       if f.lower().endswith(image_exts)]
        if not image_files:
            print(f"No image files found in {args.input}")
            return

        print(f"Found {len(image_files)} images. Processing...")
        for img_file in image_files:
            img_path = os.path.join(args.input, img_file)
            print(f"  {img_file}")
            predict_image(
                model=model,
                image_path=img_path,
                output_image_name=img_file,
                output_path=args.output_dir,
                device=device
            )
        print(f"All results saved to {args.output_dir}")

    else:
        print(f"Error: {args.input} is not a valid file or directory.")
        sys.exit(1)


# ----------------------------------------------------------------------
# Validation subcommand
# ----------------------------------------------------------------------
def add_validate_parser(subparsers):
    parser = subparsers.add_parser('validate', help='Evaluate model on validation set')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model checkpoint (.pth)')
    parser.add_argument('--images_dir', type=str, required=True,
                        help='Root directory containing "val" subfolder with images')
    parser.add_argument('--masks_dir', type=str, required=True,
                        help='Directory containing ground truth masks')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size (default: 4)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use: "cuda" or "cpu". Auto‑detect if not set.')
    parser.add_argument('--save_metrics', type=str, default=None,
                        help='Optional file path to save metrics (JSON or text)')
    return parser


def run_validate(args):
    """Run validation and compute IoU, Dice, pixel accuracy."""
    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    model = CRowNet(in_channels=3, out_channels=1)
    checkpoint = torch.load(args.model, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    print(f"Model loaded from {args.model}")

    # Setup validation dataloader
    print("\nSetting up validation dataloader...")
    val_loader = setup_val_dataloader(args.images_dir, args.masks_dir, args.batch_size)

    # Run validation (using the function from train.py)
    # We'll capture the printed output and optionally save metrics.
    print("\nRunning validation...")
    # The validate_model function in train.py prints metrics; we can also modify it to return them.
    # For now, we assume it prints. We'll create a wrapper that returns the metrics.

    # Because we don't have the original compute_* functions imported here,
    # we define them locally (copied from train.py for completeness).
    def compute_iou(outputs, targets):
        outputs = outputs > 0.5
        targets = targets > 0.5
        intersection = (outputs & targets).float().sum((1, 2, 3))
        union = (outputs | targets).float().sum((1, 2, 3))
        iou = (intersection / (union + 1e-8)).mean()
        return iou

    def compute_dice(outputs, targets):
        outputs = outputs > 0.5
        targets = targets > 0.5
        intersection = (outputs & targets).float().sum((1, 2, 3))
        dice = (2. * intersection / (outputs.float().sum((1, 2, 3)) + targets.float().sum((1, 2, 3)) + 1e-8)).mean()
        return dice

    def compute_pixel_accuracy(outputs, targets):
        correct_pixels = (outputs == targets).sum().item()
        total_pixels = targets.numel()
        return correct_pixels / total_pixels

    # Validation loop
    model.eval()
    total_iou = 0.0
    total_dice = 0.0
    total_acc = 0.0
    total_samples = 0

    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            # Apply sigmoid and threshold (same as in compute_* functions)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()

            # Compute metrics
            iou = compute_iou(preds, targets)
            dice = compute_dice(preds, targets)
            acc = compute_pixel_accuracy(preds, targets)  # returns scalar

            total_iou += iou.item() * images.size(0)
            total_dice += dice.item() * images.size(0)
            total_acc += acc * images.size(0)
            total_samples += images.size(0)

    avg_iou = total_iou / total_samples
    avg_dice = total_dice / total_samples
    avg_acc = total_acc / total_samples

    print("\n" + "="*50)
    print("Validation Results")
    print(f"  IoU              : {avg_iou:.4f}")
    print(f"  Dice Coefficient : {avg_dice:.4f}")
    print(f"  Pixel Accuracy   : {avg_acc:.4f}")
    print("="*50)

    # Optionally save metrics to file
    if args.save_metrics:
        import json
        metrics = {
            'iou': avg_iou,
            'dice': avg_dice,
            'pixel_accuracy': avg_acc,
            'samples': total_samples
        }
        with open(args.save_metrics, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to {args.save_metrics}")


# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='CRowNet: crop row detection - prediction and validation CLI'
    )
    subparsers = parser.add_subparsers(dest='command', required=True,
                                       help='Subcommand: predict or validate')

    # Add subcommands
    add_predict_parser(subparsers)
    add_validate_parser(subparsers)

    args = parser.parse_args()

    # Dispatch
    if args.command == 'predict':
        run_predict(args)
    elif args.command == 'validate':
        run_validate(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
