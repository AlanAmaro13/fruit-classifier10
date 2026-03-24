# Fruit Classification Model - Hyperparameter Search

## Overview

This notebook performs hyperparameter optimization for a Convolutional Neural Network (CNN) designed to classify fruit images into 10 categories. The optimization uses **Keras Tuner** with **Bayesian Optimization** to find the best model architecture and regularization parameters.

## Dataset Classes

The model classifies images into the following fruit categories:
- apple
- avocado
- banana
- cherry
- kiwi
- mango
- orange
- pineapple
- strawberries
- watermelon

## Input Data

- **Image dimensions**: 128×128 pixels, 3 channels (RGB)
- **Data splits**: Training, validation, and test sets loaded from pre-processed NumPy files
- **Label encoding**: String labels → integer encoding → one-hot encoding (10 classes)

## Model Architecture

The CNN architecture consists of:

### Convolutional Section
- **3 ConvBlocks** with doubling filter counts
- Kernel sizes: (8,8), (4,4), (2,2)
- AveragePooling2D for dimensionality reduction
- Batch Normalization after each convolution
- Activation: LeakyReLU

### Dense Section
- **3 fully connected layers** with halving node counts
- Dropout regularization
- Final softmax activation for multi-class classification

## Hyperparameter Search Space

| Parameter | Range | Description |
|-----------|-------|-------------|
| `Filters` | 1–50 (step 5) | Number of filters in first ConvBlock |
| `Nodes` | 10–500 (step 5) | Number of neurons in first Dense layer |
| `Dropout` | 0–50% (step 5%) | Dropout rate |
| `L1`, `L2` | 1e-6 to 1.0 | Regularization for Dense layers |
| `L1C`, `L2C` | 1e-6 to 1.0 | Regularization for Conv layers |

### Derived Parameters
- Filters progression: `[f1, 2*f1, 4*f1]`
- Nodes progression: `[n1, n1//2, n1//4]`

## Training Configuration

| Setting | Value |
|---------|-------|
| Optimizer | Adam (learning_rate=0.001) |
| Loss Function | Categorical Crossentropy |
| Batch Size | 64 |
| Epochs | 50 per trial |
| Trials | 50 |
| Executions per Trial | 2 |

### Evaluation Metrics
- Categorical Accuracy
- Precision
- Recall
- AUC

## Data Augmentation

The model incorporates **RandAugment** with:
- 2 operations per image
- Factor of 0.5
- Value range: (0, 1)

## Output Files

All results are saved to:
```
/content/drive/MyDrive/cellularAutomata/fruit_classification/notebooks/models/fruitClassifierHPSearch/
```

| File | Description |
|------|-------------|
| `best_models.txt` | Summary of best hyperparameter combinations |
| `training.log` | CSV log of training metrics |
| `*.keras` | Saved model checkpoints for best epochs |

## Optimization Objective

The Bayesian Optimization aims to **maximize validation categorical accuracy** across 50 trials.

## Usage Notes

1. **GPU recommended**: The notebook includes GPU allocation for faster training
2. **Reproducibility**: Random seed is set at the beginning of model building
3. **Custom library**: Uses a custom `AmaroXI` library for neural network components

## Dependencies

```
keras_tuner
seaborn
tensorflow
numpy
python-telegram-bot
```

