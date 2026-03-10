# Crop Row-Detection

https://doi.org/10.1016/j.compag.2024.108617

**Model Components:**
- Encoder: ResNet-34 backbone
- ASPP module for multi-scale context aggregation
- Multi-scale FPN for feature fusion
- Lightweight decoder for efficient inference

## Project Structure:
```
Row-Detection/
├── main.py                 # Main training script (runs everything)
├── train.py                # Training utilities (train_epoch, evaluate_model)
├── model.py                # CRowNet model architecture
├── dataloader.py           # Dataset and dataloader
├── utils.py                # General utilities
├── predict.py              # Inference script
├── Image/                  # Images directory
│    ├── train/
│    └── val/
├── Labels/                 # Masks directory
└── checkpoints/            # Saved models
```
# Training
```
python main.py --images_dir /data/train_images
--masks_dir /data/train_masks
--batch_size 16
--num_epochs 200
--learning_rate 1e-3
--checkpoint_dir ./output
--device cuda
```
# Prediction
## 1. On a single image
```python croprow_cli.py predict \
    --model /path/to/checkpoints/best_model.pth \
    --input /data/test_images/field1.jpg \
    --output_dir ./results \
    --device cuda
```
## 2. On a multiple images
```
python pred_val.py predict \
    --model /path/to/checkpoints/best_model.pth \
    --input /data/test_images/ \
    --output_dir ./results \
    --batch_size 4
```
# Validation
```
python pred_val.py validate \
    --model /path/to/checkpoints/best_model.pth \
    --images_dir /data/images/          # must contain 'val/' subfolder
    --masks_dir /data/masks/            # contains ground truth masks
    --batch_size 8 \
    --device cuda \
    --save_metrics ./val_metrics.json
```
# Get help
```
python pred_val.py --help
python pred_val.py predict --help
python pred_val.py validate --help
```
