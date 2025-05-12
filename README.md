# Deep Learning Assignments - IIT Jodhpur

This repository contains the implementation of various deep learning assignments completed as part of the Deep Learning course at IIT Jodhpur.

## Overview

The repository is organized into four main assignments, each focusing on different aspects of deep learning:

1. **Assignment 1**: MNIST Classification using Convolutional Neural Networks (CNNs)
   - Implementation of a CNN for digit classification
   - Class mapping and transfer learning

2. **Assignment 2**: Audio Classification using CNNs and Transformers
   - Audio data processing and feature extraction
   - Implementation of CNN and Transformer architectures for audio classification
   - Comparative analysis of different models

3. **Assignment 3**: Image Segmentation using Transfer Learning
   - Implementation of an encoder-decoder architecture
   - Fine-tuning pre-trained models for segmentation tasks
   - Performance evaluation using IoU and Dice scores

4. **Assignment 4**: Sketch-to-Image Translation using Conditional GANs
   - Implementation of conditional GANs for medical image generation
   - Label embedding for class-conditional generation
   - Evaluation using FID, Inception Score, and classifier accuracy

## Technologies Used

- Python 3.x
- PyTorch
- torchvision
- NumPy
- Matplotlib
- scikit-learn
- torchaudio
- Weights & Biases (wandb) for experiment tracking

## Repository Structure

```
DL_Assignments_IITJ/
├── Assignment1/
│   ├── d23csa001_assignment1.py
│   └── README.md
├── Assignment2/
│   ├── d23csa001_assignment2.py
│   └── README.md
├── Assignment3/
│   ├── d23csa001_assignment3.py
│   └── README.md
├── Assignment4/
│   ├── d23csa001_assignment4.py
│   └── README.md
└── README.md
```

## Getting Started

To run these assignments, you'll need to have Python and the required libraries installed. You can install the dependencies using:

```bash
pip install torch torchvision torchaudio numpy matplotlib scikit-learn wandb idx2numpy pandas torchmetrics pillow
```

Each assignment folder contains its own README with specific instructions and explanations.

## License

This project is for educational purposes only.
