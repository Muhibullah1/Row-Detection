#!/usr/bin/env python3
"""
Main training script for CRowNet crop row detection model.
Allows configuration via command‑line arguments.
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import albumentations as aug

from model import CRowNet
from dataloader import DatasetLoader
from train import train_epoch, evaluate_model


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train CRowNet for crop row detection.")
    # Paths
    parser.add_argument('--images_dir', type=str, default='Image/',
                        help='Root directory containing train/ and val/ subfolders (default: Image/)')
    parser.add_argument('--masks_dir', type=str, default='Labels/',
                        help='Directory containing ground truth masks (default: Labels/)')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory to save model checkpoints and logs (default: checkpoints)')

    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for training (default: 4)')
    parser.add_argument('--num_epochs', type=int, default=250,
                        help='Number of training epochs (default: 250)')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Initial learning rate (default: 1e-4)')

    # Device (optional override)
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use: "cuda" or "cpu". If not set, auto‑detect.')

    return parser.parse_args()


def get_augmentation_pipeline(mode='train'):
    """
    Get augmentation pipeline for training or validation.
    Args:
        mode (str): 'train' or 'val' 
    Returns:
        list: Augmentation transforms
    """
    if mode == 'train':
        return [
            aug.RandomRotate90(p=0.5),
            aug.Flip(p=0.5),
            aug.Resize(567, 567),
            aug.IAAAdditiveGaussianNoise(p=0.2),
            aug.IAAPerspective(p=0.5),
            aug.OneOf([ aug.CLAHE(p=1), aug.RandomBrightness(p=1), aug.RandomGamma(p=1)], p=0.9),
            aug.OneOf([aug.RandomContrast(p=1),aug.HueSaturationValue(p=1)], p=0.9) ]
        
    else:  # validation -> only perform resizing
        return [aug.Resize(567, 567)]


def setup_dataloaders(images_dir, masks_dir, batch_size=4):
    """
    Setup training and validation dataloaders.
    Args:
        images_dir (str): Path to images directory (contains 'train' and 'val' subfolders)
        masks_dir (str): Path to masks directory
        batch_size (int): Batch size for training
    Returns:
        tuple: (train_loader, val_loader)
    """
    # Setup paths
    train_images_dir = os.path.join(images_dir, 'train')
    val_images_dir = os.path.join(images_dir, 'val')

    # Check existence
    for d in [train_images_dir, val_images_dir, masks_dir]:
        if not os.path.isdir(d):
            raise FileNotFoundError(f"Directory not found: {d}")

    # Get image lists
    train_images = sorted(os.listdir(train_images_dir))
    val_images = sorted(os.listdir(val_images_dir))

    # Create datasets with augmentation
    train_dataset = DatasetLoader(
        train_images,
        train_images_dir,
        masks_dir,
        transform=get_augmentation_pipeline('train') )
    val_dataset = DatasetLoader(
        val_images,
        val_images_dir,
        masks_dir,
        transform=get_augmentation_pipeline('val') )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True)

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    return train_loader, val_loader


def train_model(model, train_loader, val_loader, criterion, optimizer,
                scheduler, num_epochs, device, checkpoint_dir='checkpoints'):
    """
    Train the model for specified number of epochs.
    Args:
        model: Neural network model
        train_loader: Training data loader
        val_loader: Validation data loader
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        num_epochs (int): Number of training epochs
        device: Computing device
        checkpoint_dir (str): Directory to save model checkpoints
    Returns:
        tuple: (training_losses, validation_losses)
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 60)

        # Training phase
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        print(f"Training Loss: {train_loss:.4f}")

        # Validation phase
        val_loss = evaluate_model(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        print(f"Validation Loss: {val_loss:.4f}")

        # Learning rate scheduling
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Learning Rate: {current_lr:.6f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, checkpoint_path)
            print(f"✓ Best model saved (Val Loss: {val_loss:.4f})")

        # Save periodic checkpoint
        if (epoch + 1) % 50 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'model_epoch_{epoch + 1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, checkpoint_path)
            print(f"Checkpoint saved at epoch {epoch + 1}")

    return train_losses, val_losses


def main():
    """Main training function with command‑line arguments."""
    args = parse_args()

    # Device setup
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Log configuration
    print("\nTraining configuration:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")

    # Setup dataloaders
    print("\nSetting up dataloaders...")
    train_loader, val_loader = setup_dataloaders(
        args.images_dir,
        args.masks_dir,
        args.batch_size)

    # Initialize model
    print("\nInitializing model...")
    model = CRowNet(in_channels=3, out_channels=1)
    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Setup training components
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.2, patience=5, verbose=True )

    # Train model
    print("\nStarting training...")
    train_losses, val_losses = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=args.num_epochs,
        device=device,
        checkpoint_dir=args.checkpoint_dir)

    # Save final training history
    history_path = os.path.join(args.checkpoint_dir, 'training_history.npz')
    np.savez(history_path, train_loss=train_losses, val_loss=val_losses)
    print(f"\nTraining history saved to {history_path}")
    print("Training completed!")


if __name__ == "__main__":
    main()
