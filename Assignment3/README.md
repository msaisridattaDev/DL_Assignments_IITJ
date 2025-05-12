# Assignment 3: Image Segmentation using Transfer Learning

## Overview

This assignment focuses on implementing image segmentation using transfer learning. The project uses a pre-trained MobileNetV2 model as an encoder and a custom decoder for semantic segmentation tasks. Two approaches are compared:

1. **Feature Extraction**: Using a pre-trained encoder with frozen weights
2. **Fine-Tuning**: Fine-tuning both the encoder and decoder together

## Dataset

The dataset consists of RGB images and their corresponding binary segmentation masks. The data is organized into:
- Training set: Images in the "train" folder and masks in the "train_masks" folder
- Test set: Images in the "test" folder and masks in the "test_masks" folder

The images are resized to 128×128 pixels for both training and testing.

## Implementation Details

### Data Processing

- Images and masks are loaded and resized to 128×128 pixels
- Data augmentation techniques are commented out but available (ColorJitter, RandomRotation, etc.)
- Custom PyTorch Dataset class for handling image-mask pairs

### Model Architecture

#### Encoder
- Pre-trained MobileNetV2 model with the classification head removed
- In the feature extraction approach, the encoder weights are frozen
- In the fine-tuning approach, the encoder weights are trainable

#### Custom Decoder
```python
class Custom_Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Custom_Decoder, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        self.upconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.upconv5 = nn.ConvTranspose2d(64, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
```

The decoder consists of:
- 5 convolutional layers for downsampling
- 5 transposed convolutional layers for upsampling
- ReLU activation functions
- Sigmoid activation for the final output layer
- Xavier initialization for all convolutional weights

### Training and Evaluation

- Binary Cross-Entropy Loss (BCELoss)
- Adam optimizer with different learning rates for feature extraction and fine-tuning
- 30 epochs for both approaches
- Evaluation metrics: IoU (Intersection over Union) and Dice Score

## Results

### Feature Extraction Approach
- Training Loss: Decreases from ~0.7 to ~0.3
- Testing Loss: Decreases from ~0.7 to ~0.35
- IoU: ~0.7
- Dice Score: ~0.8

### Fine-Tuning Approach
- Training Loss: Decreases from ~0.6 to ~0.25
- Testing Loss: Decreases from ~0.6 to ~0.3
- IoU: ~0.75
- Dice Score: ~0.85

The fine-tuning approach shows better performance in terms of all metrics, demonstrating the benefit of adapting the pre-trained encoder to the specific segmentation task.

## Visualizations

The code includes visualizations of:
- Original images, generated masks, and ground truth masks
- Training and testing loss curves for both approaches

## How to Run

To run the code:

```python
python d23csa001_assignment3.py
```

Note: You may need to modify the file paths for the dataset in the code.

## Dependencies

- PyTorch
- torchvision
- NumPy
- Matplotlib
- PIL (Python Imaging Library)

## Key Learnings

- Implementation of encoder-decoder architectures for image segmentation
- Transfer learning techniques: feature extraction and fine-tuning
- Custom decoder design for segmentation tasks
- Evaluation metrics for segmentation: IoU and Dice Score
- Visualization of segmentation results
