# Fruit Classification CNN Model

A convolutional neural network (CNN) for fruit classification using TensorFlow/Keras. This model classifies images of fruits into 10 different categories.

## Overview

This project implements a CNN for fruit image classification. The model is built using TensorFlow/Keras with custom modules from the AmaroXI library, providing a modular architecture for easy experimentation and hyperparameter tuning.

## Dataset

The model works with 10 fruit classes:
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

### Data Format
- **Image size**: 128 × 128 pixels
- **Channels**: 3 (RGB)
- **Data split**: Training, validation, and test sets
- **Labels**: String labels converted to one-hot encoding (10 classes)

## Model Architecture

### Convolutional Neural Network Structure

| Layer Type | Details |
|------------|---------|
| **Input** | 128 × 128 × 3 |
| **Conv Block 1** | 25 filters, 11×11 kernel |
| **Pooling** | 2×2 Average Pooling |
| **Conv Block 2** | 50 filters, 5×5 kernel |
| **Pooling** | 2×2 Average Pooling |
| **Conv Block 3** | 100 filters, 3×3 kernel |
| **Pooling** | 2×2 Average Pooling |
| **Flatten** | - |
| **Dense Layer** | 50 neurons with LeakyReLU |
| **Dense Layer** | 25 neurons with LeakyReLU |
| **Output Layer** | 10 neurons with Softmax |

### Key Features
- **Regularization**: L1L2 regularization (1e-6 each) and 5% dropout
- **Weight Initialization**: He Normal
- **Batch Normalization**: Applied after each convolutional layer
- **Activation Functions**: LeakyReLU for hidden layers, Softmax for output

## Training Configuration

### Compilation Settings
- **Optimizer**: Adam (learning rate = 0.001)
- **Loss Function**: Categorical Crossentropy
- **Metrics**: AUC, Precision, Recall, Categorical Accuracy
- **Batch Size**: 64
- **Epochs**: 500 

### Callbacks
- **Early Stopping**: Patience of 1000 epochs (not used), restores best weights (used)
- **Reduce LR on Plateau**: Factor 0.8, patience 1000, min_lr 1e-6 (not used)
- **Model Checkpoint**: Saves best models based on multiple metrics:
  - Validation loss (min)
  - Categorical accuracy (max)
  - Validation precision (max)
  - Validation recall (max)
  - Validation AUC (max)
- **CSV Logger**: Records training history
- **TensorBoard**: Logs for visualization

## Evaluation Metrics

The model includes comprehensive evaluation tools through the `MultiClassEvaluator` class, providing:

### Visualizations
- Confusion Matrix (raw counts and normalized percentages)
- Per-class metrics (precision, recall, F1-score)
- ROC curves (One-vs-Rest)
- Precision-Recall curves
- Top-k accuracy plots
- Class distribution analysis
- Calibration curves (reliability diagrams)

### Performance Metrics
- Overall accuracy
- Macro and weighted averages
- Per-class support counts
- Micro/macro AUC scores
- Top-1, Top-2, Top-3, and Top-5 accuracy

## Usage

### Loading Data
```python
# Load preprocessed data
x_train = np.load('/path/to/x_train.npy')
x_test = np.load('/path/to/x_test.npy')
x_val = np.load('/path/to/x_val.npy')

y_train = np.load('/path/to/y_train.npy')
y_test = np.load('/path/to/y_test.npy')
y_val = np.load('/path/to/y_val.npy')
```

### Training the Model
```python
model_CNN = CNN_Model()
model_CNN.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=[keras.metrics.AUC(), keras.metrics.Precision(),
             keras.metrics.Recall(), keras.metrics.CategoricalAccuracy()]
)

model_trained = model_CNN.fit(
    x=x_train, y=y_train_oh,
    epochs=500, batch_size=64,
    validation_data=(x_val, y_val_oh),
    callbacks=callbacks
)
```

### Evaluating the Model
```python
# Predict on test set
y_preds = model_CNN.predict(x_test)
y_preds_int = np.argmax(y_preds, axis=1)

# Generate evaluation report
evaluator = MultiClassEvaluator(
    y_test_int, y_preds_int, y_preds,
    class_names=['apple', 'avocado', ...]
)
evaluator.generate_complete_report()
```

## File Structure

```
fruit_classification/
├── notebooks/
│   ├── models/
│   │   └── fruitClassifierTest/
│   │       ├── model.h5
│   │       ├── categorical_accuracy_max.keras
│   │       ├── training.log
│   │       ├── images/
│   │       └── logs/          # TensorBoard logs
│   └── AmaroXI/
│       └── AmaroX/
│           ├── ai_functions.py
│           ├── Convolutional.py
│           ├── data_manipulation.py
│           ├── utilities.py
│           └── DNN.py
└── database/
    ├── x_train.npy
    ├── x_test.npy
    ├── x_val.npy
    ├── y_train.npy
    ├── y_test.npy
    └── y_val.npy
```

## Visualization Settings

The notebook includes extensive matplotlib/Seaborn configuration for high-quality figures:
- DPI: 300 for publication-ready images
- Custom font sizes and weights
- Grid styling and tick configurations

## Requirements

- Python 3.x
- TensorFlow 2.x
- NumPy
- Matplotlib
- Seaborn
- scikit-learn
- pandas

## Key Functions

| Function | Description |
|----------|-------------|
| `G_ConvBlock_2D()` | Creates a convolutional block with Conv2D → BatchNorm → Activation → Pooling |
| `CNN_2D()` | Builds the complete CNN architecture with configurable layers |
| `standard_callbacks()` | Sets up training callbacks with flexible monitoring |
| `MultiClassEvaluator` | Comprehensive evaluation class for multi-class classification |

## Notes

- The model uses GPU acceleration when available
- Random seed is set for reproducibility
- Model checkpoints are saved for each monitored metric
- The architecture is designed to preserve spatial dimensions through convolutional layers while reducing via pooling

