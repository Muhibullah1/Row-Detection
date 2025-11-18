# Crop Row-Detection

https://doi.org/10.1016/j.compag.2024.108617

**Model Components:**
- Encoder: ResNet-34 backbone
- ASPP module for multi-scale context aggregation
- Multi-scale FPN for feature fusion
- Lightweight decoder for efficient inference

## Project Structure:
crop_row_detection/

├── main.py                 # Main training script (runs everything)

├── train.py                # Training utilities (train_epoch, evaluate_model)

├── model.py                # CRowNet model architecture

├── dataloader.py           # Dataset and dataloader

├── utils.py                # General utilities

├── predict.py              # Inference script

├── Image/                  # Images directory

│   ├── train/

│   └── val/

├── Labels/                 # Masks directory

└── checkpoints/            # Saved models

# Training
python main.py
