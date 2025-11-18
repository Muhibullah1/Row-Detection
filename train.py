
import cv2
import os
import sys
import time
import torch
import numpy as np
import torch.optim as optim
import albumentations as aug
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

os.environ["CUDA_VISIBLE_DEVICES"]="1"


def validate_model(model, dataloader, device):
    """ Set model to evaluation mode """
  
    model.eval()
    # Initialize variables to store metrics
    total_iou = 0.0
    total_dice = 0.0
    total_pixel_accuracy = 0.0
    total_samples = 0
    
    with torch.no_grad():
      
        for images, targets in dataloader:
            # Move data to the device
            images = images.to(device)
            targets = targets.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Compute segmentation metrics
            iou = compute_iou(outputs, targets)
            dice = compute_dice(outputs, targets)
            pixel_accuracy = compute_pixel_accuracy(outputs, targets)
            
            # Aggregate metrics
            total_iou += iou.item() * images.size(0)
            total_dice += dice.item() * images.size(0)
            total_pixel_accuracy += pixel_accuracy * images.size(0)
            total_samples += images.size(0)
    
    # Compute average metrics
    avg_iou = total_iou / total_samples
    avg_dice = total_dice / total_samples
    avg_pixel_accuracy = total_pixel_accuracy / total_samples
    
    print(f'Validation IOU: {avg_iou}, Dice Coefficient: {avg_dice}, Pixel Accuracy: {avg_pixel_accuracy}')

def compute_iou(outputs, targets):
    """ Threshold outputs if they are logits or probabilities """
  
    outputs = outputs > 0.5
    targets = targets > 0.5

    # Compute Intersection over Union (IoU)
    intersection = (outputs & targets).float().sum((1, 2, 3))  # include batch dim
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
    """ Compute pixel accuracy """
  
    correct_pixels = (outputs == targets).sum().item()
    total_pixels = targets.numel()
    pixel_accuracy = correct_pixels / total_pixels
    return pixel_accuracy

    
def train_epoch(model, train_loader, criterion,optimizer, device):
  
    model.train()
    model.to(device)
    running_loss = 0.0
    dice_coef = 0.0
    #start_time = time.time()
    for ip,op in tqdm_notebook(train_loader): 
        #print(ip.shape)
        optimizer.zero_grad()   # .backward() accumulates gradients
        data = ip.to(device)
        target = op.to(device) # all data & model on same device
        outputs = model(data)
        loss = criterion(outputs, target)
        running_loss += loss.item()

        loss.backward()
        optimizer.step()
    
    #end_time = time.time()
    
    running_loss /= len(train_loader)
    print('Training Loss: ', running_loss)#, 'Time: ',end_time - start_time, 's')
    return running_loss    


def eval_model(model, test_loader, criterion, device):
  
    with torch.no_grad():
        model.eval()
        model.to(device)
        running_loss = 0.0
        dice_coef = 0.0
      
        #start_time = time.time()
        for ip,op in tqdm_notebook(test_loader):   
            data = ip.to(device)
            target = op.to(device)
            outputs = model(data)
            loss = criterion(outputs, target).detach()
            running_loss += loss.item()
        #end_time = time.time()
        running_loss /= len(test_loader)
        print('Testing Loss: ', running_loss)#, 'Time: ',end_time - start_time, 's')
        return running_loss
        
