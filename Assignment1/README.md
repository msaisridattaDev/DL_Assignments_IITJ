# Assignment 1: MNIST Classification using Convolutional Neural Networks

## Overview

This assignment focuses on implementing Convolutional Neural Networks (CNNs) for classifying handwritten digits from the MNIST dataset. The assignment is divided into two main tasks:

1. **Task 1**: Standard MNIST Classification (10 classes)
2. **Task 2**: Modified MNIST Classification with class mapping (4 classes)

## Dataset

The MNIST dataset consists of 28x28 grayscale images of handwritten digits (0-9). The dataset is split into:
- Training set: 60,000 examples
- Test set: 10,000 examples

## Implementation Details

### CNN Architecture

The CNN architecture used for both tasks consists of:

1. **Convolutional Layers**:
   - Conv1: 1 → 16 channels, 7×7 kernel, stride 1, padding 3, followed by ReLU and MaxPool
   - Conv2: 16 → 8 channels, 5×5 kernel, stride 1, padding 2, followed by ReLU and MaxPool
   - Conv3: 8 → 4 channels, 3×3 kernel, stride 2, padding 1, followed by ReLU and AvgPool

2. **Fully Connected Layers**:
   - FC1: 4×2×2 → 64 neurons with ReLU
   - FC2: 64 → 32 neurons with ReLU
   - FC3: 32 → 10 neurons (for Task 1) or 32 → 10 → 4 neurons (for Task 2)

### Task 1: Standard MNIST Classification

In this task, the CNN is trained to classify the digits into their original 10 classes (0-9). The model is trained for 10 epochs using the Adam optimizer with a learning rate of 0.003 and Cross-Entropy Loss.

Key components:
- Data preprocessing: Normalization (dividing by 255)
- Train-validation split: 80%-20%
- Evaluation metrics: Accuracy and confusion matrix

### Task 2: Modified MNIST Classification

In this task, the original 10 classes are mapped to 4 new classes according to the following mapping:
```
Class Mapping:
0 → 0
1 → 1
2 → 2
3 → 2
4 → 3
5 → 2
6 → 0
7 → 1
8 → 2
9 → 3
```

The same CNN architecture is used but with an additional fully connected layer to output 4 classes instead of 10.

## Results

### Task 1: Standard MNIST Classification
- Training accuracy: ~98%
- Validation accuracy: ~97%
- Test accuracy: ~97%

### Task 2: Modified MNIST Classification
- Training accuracy: ~99%
- Validation accuracy: ~98%
- Test accuracy: ~98%

## Visualizations

The code includes visualizations of:
- Training and validation loss curves
- Validation accuracy curves
- Test accuracy curves
- Confusion matrices for both tasks

## How to Run

To run the code:

```python
python d23csa001_assignment1.py
```

Note: You may need to modify the file paths for the MNIST dataset in the code.

## Dependencies

- PyTorch
- NumPy
- Matplotlib
- scikit-learn
- idx2numpy

## Key Learnings

- Implementation of CNNs for image classification
- Data preprocessing and normalization techniques
- Model evaluation and performance visualization
- Transfer learning and class mapping strategies
