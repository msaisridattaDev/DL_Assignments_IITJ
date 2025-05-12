# Assignment 2: Audio Classification using CNNs and Transformers

## Overview

This assignment focuses on implementing and comparing different deep learning architectures for audio classification. The project uses the ESC-10 dataset (Environmental Sound Classification) and implements both Convolutional Neural Networks (CNNs) and CNN-Transformer hybrid models for audio classification.

## Dataset

The ESC-10 dataset consists of 400 environmental audio recordings (10 classes with 40 samples each) organized into 5 folds for cross-validation. The classes include:

- Dog bark
- Rain
- Sea waves
- Baby cry
- Clock tick
- Person sneeze
- Helicopter
- Chainsaw
- Rooster
- Fire crackling

The dataset is already pre-divided into 5 folds, with the first fold used for testing and the remaining folds used for 4-fold validation.

## Implementation Details

### Data Processing

- Audio files are loaded and resampled from 44.1kHz to 16kHz
- Audio samples are split into overlapping windows
- Custom PyTorch Dataset and DataLoader classes are implemented for efficient data handling

### Model Architectures

#### 1. CNN Model

The CNN architecture consists of:
- 3 convolutional blocks, each with Conv1D, ReLU, and pooling layers
- Fully connected layers for classification
- Trained with Adam optimizer and Cross-Entropy Loss

```python
class CNN(nn.Module):
  def __init__(self):
    super(CNN,self).__init__()

    self.convolution1 = nn.Sequential(
        nn.Conv1d(in_channels=1,out_channels=16,kernel_size=3,stride=1,padding=3),
        nn.ReLU(),
        nn.MaxPool1d(kernel_size=2,stride=2)
    )

    self.convolution2 = nn.Sequential(
        nn.Conv1d(in_channels=16,out_channels=8,kernel_size=3,stride=1,padding=2),
        nn.ReLU(),
        nn.MaxPool1d(kernel_size=2,stride=2)
    )

    self.convolution3 = nn.Sequential(
        nn.Conv1d(in_channels=8,out_channels=4,kernel_size=3,stride=2,padding=1),
        nn.ReLU(),
        nn.MaxPool1d(kernel_size=2,stride=2)
    )

    self.fully_connected_layer1 = nn.Linear(1000*4,32)
    self.fully_connected_layer2 = nn.Linear(32,10)
```

#### 2. CNN-Transformer Hybrid Models

Three variants of CNN-Transformer hybrid models are implemented with different numbers of attention heads:
- Model with 1 attention head
- Model with 2 attention heads
- Model with 4 attention heads

The architecture consists of:
- CNN feature extractor
- Transformer encoder with positional encoding
- Classification head

Key components include:
- Custom positional encoding
- Multi-head self-attention mechanism
- Feed-forward networks
- Layer normalization and residual connections

### Training and Evaluation

- Models are trained for 100 epochs
- Weights & Biases (wandb) is used for experiment tracking
- Performance metrics include accuracy, F1 score, confusion matrix, and AUC-ROC scores

## Results

### CNN Model
- Training accuracy: ~85%
- Test accuracy: ~80%

### CNN-Transformer Models
- 1-head model: Test accuracy ~82%
- 2-head model: Test accuracy ~83%
- 4-head model: Test accuracy ~85%

The results show that the CNN-Transformer hybrid models, especially with more attention heads, outperform the pure CNN model for audio classification.

## Visualizations

The code includes visualizations of:
- Training and validation loss curves
- Accuracy curves
- Confusion matrices
- F1 scores by class
- AUC-ROC scores for each class

## How to Run

To run the code:

```python
python d23csa001_assignment2.py
```

Note: You may need to modify the file paths for the ESC-10 dataset in the code.

## Dependencies

- PyTorch
- torchaudio
- NumPy
- Matplotlib
- scikit-learn
- Weights & Biases (wandb)
- pytorch_lightning

## Key Learnings

- Audio data processing and feature extraction
- Implementation of CNN architectures for audio classification
- Implementation of Transformer architectures for sequence modeling
- Comparative analysis of different model architectures
- Experiment tracking and visualization with wandb
