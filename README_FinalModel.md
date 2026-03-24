# Fruit Classification CNN Model

A convolutional neural network (CNN) for multi-class fruit classification, featuring comprehensive data augmentation, model training with callbacks, and extensive evaluation metrics.

## Overview

This project implements a deep learning model to classify 10 different types of fruits from images. The model uses a custom CNN architecture with data augmentation to achieve robust classification performance.

### Supported Classes
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

## Model Architecture

The CNN architecture consists of:
- **3 Convolutional Blocks** with increasing filter sizes (46, 92, 184)
- Kernel sizes: (8,8), (4,4), (2,2)
- Average Pooling layers with (2,2) pooling and stride
- **3 Dense Layers** (500, 250, 125 units) with 50% dropout
- Softmax output layer for 10-class classification

### Key Features
- L1L2 regularization for both convolutional and dense layers
- He normal weight initialization
- LeakyReLU activation in hidden layers
- Softmax for final classification

## Data Augmentation

Advanced augmentation techniques using Albumentations library:

### Transformations Applied
1. **Geometric**: Horizontal/Vertical flip, Random rotation, ShiftScale rotation
2. **Blur & Noise**: Gaussian blur, Gaussian noise, ISO noise
3. **Color**: Brightness/contrast adjustment, Hue/Saturation/Value shifts, RGB shift
4. **Spatial**: Random cropping/resizing, Elastic transformations, Grid distortion
5. **Occlusion**: Coarse dropout (cutout)
6. **Affine**: Translation, rotation, scaling

The pipeline augments the original dataset from ~2,000 to 50,000 images.

## Training Configuration

### Hyperparameters
- **Optimizer**: Adam (learning rate: 0.001)
- **Loss**: Categorical Crossentropy
- **Metrics**: AUC, Precision, Recall, Categorical Accuracy
- **Batch Size**: 256
- **Epochs**: 50

### Callbacks
- EarlyStopping (restores best weights)
- ModelCheckpoint (saves best models for each metric)
- CSVLogger
- TensorBoard

## Usage

### Dependencies
```bash
pip install tensorflow numpy seaborn matplotlib scikit-learn albumentations opencv-python
```

### Loading Data
```python
x_train = np.load('path/to/x_train.npy')
x_test = np.load('path/to/x_test.npy')
x_val = np.load('path/to/x_val.npy')
y_train = np.load('path/to/y_train.npy')
y_test = np.load('path/to/y_test.npy')
y_val = np.load('path/to/y_val.npy')
```

### Creating and Training Model
```python
# Create model
model = CNN_Model()
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['AUC', 'Precision', 'Recall', 'CategoricalAccuracy'])

# Train model
history = model.fit(x_train, y_train,
                    validation_data=(x_val, y_val),
                    epochs=50,
                    batch_size=256,
                    callbacks=callbacks)
```

### Evaluation
```python
# Predict and evaluate
y_preds = model.predict(x_test)
y_preds_int = np.argmax(y_preds, axis=1)

# Generate comprehensive report
evaluator = MultiClassEvaluator(y_test_int, y_preds_int, y_preds, class_names)
evaluator.generate_complete_report()
```

## Evaluation Tools

The `MultiClassEvaluator` class provides comprehensive evaluation capabilities:

### Metrics Visualizations
- Confusion matrix (raw counts and percentages)
- Per-class precision, recall, and F1-score
- ROC curves (One-vs-Rest)
- Precision-Recall curves
- Top-k accuracy
- Class distribution analysis
- Calibration curves (reliability diagrams)

### Performance Metrics
- Accuracy
- Precision (macro/weighted)
- Recall (macro/weighted)
- F1-score (macro/weighted)
- AUC (micro/macro)
- Top-k accuracy (k=1,2,3,5)

## Results

Based on the evaluation output:
- **Overall Accuracy**: 73.75%
- **Macro Average**: Precision 77%, Recall 74%, F1-score 74%
- **Best Performing Classes**: Strawberries (100% precision), Kiwi (89% precision), Apple (85% precision)
- **Areas for Improvement**: Mango (48% precision, 69% recall), Banana (83% precision, 49% recall)

## Project Structure

```
fruit_classification/
├── notebooks/
│   ├── models/
│   │   └── fruitClassifierCompleted/
│   │       ├── model.h5
│   │       ├── categorical_accuracy_max.keras
│   │       ├── training.log
│   │       ├── logs/
│   │       └── images/
│   └── AmaroXI/AmaroX/
│       ├── ai_functions.py
│       ├── Convolutional.py
│       ├── data_manipulation.py
│       └── DNN.py
└── database/
    ├── x_train.npy
    ├── x_test.npy
    ├── x_val.npy
    ├── y_train.npy
    ├── y_test.npy
    └── y_val.npy
```

## Model Saving

The model is saved in multiple formats:
- `model.h5`: Complete model architecture and weights
- `categorical_accuracy_max.keras`: Best model based on validation categorical accuracy
- Training logs and TensorBoard files for performance monitoring

## Notes

- The model uses GPU acceleration when available
- Data augmentation generates up to 50,000 training samples
- Callbacks monitor multiple metrics (loss, accuracy, precision, recall, AUC)
- Best model weights are automatically restored after early stopping
