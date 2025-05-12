# Assignment 4: Sketch-to-Image Translation using Conditional GANs

## Overview

This assignment focuses on implementing a Conditional Generative Adversarial Network (cGAN) for sketch-to-image translation. The project uses a dataset of medical images (ISIC skin lesion dataset) and their corresponding sketches/contours to generate realistic images from sketch inputs. The implementation also includes evaluation metrics to assess the quality of the generated images.

## Dataset

The dataset consists of:
- Medical images (ISIC skin lesion dataset)
- Corresponding sketch/contour images
- Class labels for each image

The data is organized into:
- Training set: Images in "Train/Train_data/", sketches in "Train/Train_contours/", and labels in "Train/Train_labels.csv"
- Test set: Images in "Test/Test_data/", sketches in "Test/Test_contours/", and labels in "Test/Test_labels.csv"

## Implementation Details

### Data Processing

- Images and sketches are resized to 256×256 pixels
- Images are normalized to the range [-1, 1]
- Labels are one-hot encoded and then converted to class indices (0-6)
- Custom data loading and preprocessing functions for both images and sketches

### Model Architecture

#### Generator
The generator follows a U-Net-like architecture with:
- Label embedding layer to incorporate class information
- Encoder (downsampling) path with convolutional layers
- Decoder (upsampling) path with transposed convolutional layers
- Skip connections between encoder and decoder layers
- Tanh activation for the output layer

```python
class Generator(nn.Module):
    def __init__(self, n_classes, image_channels=1):
        super(Generator, self).__init__()
        self.label_embedding = LabelEmbedding(n_classes)
        
        # Process combined image and label embedding
        self.conv1 = nn.Conv2d(image_channels + 1, 64, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        
        self.down_stack = nn.ModuleList([
            downsample(64, 128, apply_batchnorm=False),
            downsample(128, 256),
            downsample(256, 512),
            downsample(512, 512),
            downsample(512, 512),
            downsample(512, 512),
            downsample(512, 512),
        ])
        
        self.up_stack = nn.ModuleList([
            upsample(512, 512, apply_dropout=True),
            upsample(1024, 512, apply_dropout=True),
            upsample(1024, 512, apply_dropout=True),
            upsample(1024, 512),
            upsample(1024, 256),
            upsample(512, 128),
            upsample(256, 64),
        ])
        
        # Final output layer
        self.last_1 = nn.ConvTranspose2d(256, 3, kernel_size=4, stride=2, padding=1)
        self.last_2 = nn.ConvTranspose2d(3, 3, kernel_size=4, stride=2, padding=1)
```

#### Discriminator
The discriminator follows a PatchGAN architecture:
- Label embedding layer to incorporate class information
- Series of downsampling convolutional layers
- Final convolutional layer to produce a patch of outputs
- No sigmoid activation (uses BCE with logits loss)

```python
class Discriminator(nn.Module):
    def __init__(self, n_classes):
        super(Discriminator, self).__init__()
        
        self.label_embedding = LabelEmbedding(n_classes)
        
        self.down1 = downsample(4, 64, apply_batchnorm=False)
        self.down2 = downsample(64, 128)
        self.down3 = downsample(128, 256)
        
        self.zero_pad1 = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=0, bias=False)
        self.batchnorm1 = nn.BatchNorm2d(512)
        self.leaky_relu = nn.LeakyReLU(0.2)
        
        self.zero_pad2 = nn.ZeroPad2d(1)
        self.last = nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0)
```

### Loss Functions

- **Generator Loss**: Combination of adversarial loss (BCE with logits) and L1 loss (pixel-wise difference)
- **Discriminator Loss**: Binary cross-entropy with logits loss for real and fake images

### Training and Evaluation

- Adam optimizer with learning rate 0.0002 and betas (0.5, 0.999)
- Training for multiple epochs with batch size 64
- Weights & Biases (wandb) for experiment tracking (commented out in the code)
- Evaluation metrics:
  - FID (Fréchet Inception Distance) for measuring the quality of generated images
  - Inception Score for assessing diversity and quality
  - Classifier accuracy to compare real and generated images

## Results

The model is evaluated using:

1. **Loss Curves**:
   - Generator total loss, GAN loss, and L1 loss
   - Discriminator loss

2. **Image Quality Metrics**:
   - FID Score: Measures the distance between the feature distributions of real and generated images
   - Inception Score: Assesses the quality and diversity of generated images

3. **Classification Accuracy**:
   - A ResNet-18 classifier is trained on the original images
   - The classifier is then used to evaluate both real test images and generated images
   - Comparison of accuracy between real and generated images indicates how well the generator preserves class-specific features

## Visualizations

The code includes visualizations of:
- Original images and their labels
- Sketch images
- Generated images from sketches
- Loss curves for generator and discriminator

## How to Run

To run the code:

```python
python d23csa001_assignment4.py
```

Note: You may need to modify the file paths for the dataset in the code.

## Dependencies

- PyTorch
- torchvision
- torchmetrics
- NumPy
- Matplotlib
- pandas
- PIL (Python Imaging Library)
- scikit-learn
- Weights & Biases (wandb) - optional for experiment tracking

## Key Learnings

- Implementation of conditional GANs for image-to-image translation
- Label embedding techniques for conditional generation
- U-Net architecture with skip connections for image generation
- Evaluation metrics for generative models (FID, Inception Score)
- Classifier-based evaluation of generated images
