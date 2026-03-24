# Fruit Classification - Complete Project Documentation

A comprehensive deep learning project for multi-class fruit classification, covering data preprocessing, hyperparameter optimization, model training, and evaluation.

## Project Overview

This project implements a complete pipeline for classifying 10 different fruit types using Convolutional Neural Networks (CNNs). It includes data acquisition, preprocessing, hyperparameter tuning, model training with augmentation, and comprehensive evaluation.

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

### Project Structure
```
fruit_classification/
├── database/                      # Preprocessed data
│   ├── x_train.npy
│   ├── x_test.npy
│   ├── x_val.npy
│   ├── y_train.npy
│   ├── y_test.npy
│   └── y_val.npy
├── notebooks/
│   ├── models/
│   │   ├── fruitClassifierCompleted/   # Final trained model
│   │   ├── fruitClassifierHPSearch/    # Hyperparameter search results
│   │   └── fruitClassifierTest/        # Test model
│   └── AmaroXI/AmaroX/                 # Custom utility modules
│       ├── ai_functions.py
│       ├── Convolutional.py
│       ├── data_manipulation.py
│       └── DNN.py
└── README_*.md                   # Detailed documentation per stage
```

## 1. Data Acquisition & Preprocessing

### Dataset Source
[Fruit Classification (10-Class) Dataset](https://www.kaggle.com/datasets/karimabdulnabi/fruit-classification10-class) from Kaggle

### Preprocessing Steps
1. **Data Acquisition**: Automatic download via Kaggle Hub
2. **Exploratory Analysis**: Resolution distribution, class balance, file format verification
3. **Image Processing**: Resizing to 128×128 pixels, normalization to [0,1] range
4. **Data Splitting**: Stratified split (70% train, 15% validation, 15% test)
5. **Export**: NumPy arrays (.npy) for efficient loading

### Key Statistics
- **Total images**: 3,374
- **Format**: JPEG
- **Input shape**: 128 × 128 × 3 (RGB)

---

## 2. Hyperparameter Search

### Optimization Method
- **Framework**: Keras Tuner with Bayesian Optimization
- **Trials**: 50
- **Executions per trial**: 2
- **Target**: Maximize validation categorical accuracy

### Search Space

| Parameter | Range | Description |
|-----------|-------|-------------|
| `Filters` | 1–50 (step 5) | Initial ConvBlock filters |
| `Nodes` | 10–500 (step 5) | First Dense layer neurons |
| `Dropout` | 0–50% (step 5%) | Dropout rate |
| `L1`, `L2` | 1e-6 to 1.0 | Dense layer regularization |
| `L1C`, `L2C` | 1e-6 to 1.0 | Conv layer regularization |

### Derived Architecture
- Filters progression: `[f1, 2*f1, 4*f1]`
- Nodes progression: `[n1, n1//2, n1//4]`

### Data Augmentation
- **RandAugment**: 2 operations per image, factor 0.5

---

## 3. Model Architecture

### Final Model Configuration

| Component | Details |
|-----------|---------|
| **Input** | 128 × 128 × 3 |
| **Conv Block 1** | 46 filters, 8×8 kernel + AvgPooling |
| **Conv Block 2** | 92 filters, 4×4 kernel + AvgPooling |
| **Conv Block 3** | 184 filters, 2×2 kernel + AvgPooling |
| **Dense Layer 1** | 500 units + 50% dropout |
| **Dense Layer 2** | 250 units + 50% dropout |
| **Dense Layer 3** | 125 units + 50% dropout |
| **Output** | 10 units, Softmax activation |

### Key Features
- **Regularization**: L1L2 regularization on all layers
- **Weight Initialization**: He Normal
- **Activation**: LeakyReLU in hidden layers
- **Normalization**: Batch Normalization after convolutions

---

## 4. Data Augmentation (Final Model)

Advanced augmentation using **Albumentations**:

| Category | Transformations |
|----------|-----------------|
| **Geometric** | Horizontal/Vertical flip, Random rotation, ShiftScale rotation |
| **Blur & Noise** | Gaussian blur, Gaussian noise, ISO noise |
| **Color** | Brightness/contrast, HSV shift, RGB shift |
| **Spatial** | Random crop/resize, Elastic transform, Grid distortion |
| **Occlusion** | Coarse dropout (cutout) |

**Result**: Augments dataset from ~2,000 to 50,000 training images.

---

## 5. Training Configuration

### Hyperparameters
| Setting | Value |
|---------|-------|
| Optimizer | Adam (lr=0.001) |
| Loss | Categorical Crossentropy |
| Batch Size | 256 (Final) / 64 (Search) |
| Epochs | 50 (Final) / 50 per trial (Search) |

### Callbacks
- **ModelCheckpoint**: Saves best models per metric
- **CSVLogger**: Records training history
- **TensorBoard**: Visualization logs

### Metrics Tracked
- Categorical Accuracy
- AUC
- Precision
- Recall

---

## 6. Evaluation Tools

### MultiClassEvaluator Class Features

| Visualization | Description |
|---------------|-------------|
| Confusion Matrix | Raw counts and normalized percentages |
| Per-Class Metrics | Precision, recall, F1-score table |
| ROC Curves | One-vs-Rest for each class |
| Precision-Recall Curves | Class-wise PR curves |
| Top-k Accuracy | k=1,2,3,5 |
| Class Distribution | Sample distribution analysis |
| Calibration Curves | Reliability diagrams |

### Performance Metrics
- Overall accuracy
- Macro/Weighted averages (precision, recall, F1)
- Micro/Macro AUC
- Per-class support counts

---

## 7. Results (Final Model)

| Metric | Score |
|--------|-------|
| **Overall Accuracy** | 73.75% |
| **Macro Precision** | 77% |
| **Macro Recall** | 74% |
| **Macro F1-Score** | 74% |

### Per-Class Performance

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Strawberries | 100% | - | - |
| Kiwi | 89% | - | - |
| Apple | 85% | - | - |
| Banana | 83% | 49% | - |
| Mango | 48% | 69% | - |

---

## 8. Installation & Usage

### Dependencies
```bash
pip install tensorflow numpy seaborn matplotlib scikit-learn albumentations opencv-python keras-tuner kagglehub tqdm
```

### Loading Preprocessed Data
```python
import numpy as np

x_train = np.load('database/x_train.npy')
x_test = np.load('database/x_test.npy')
x_val = np.load('database/x_val.npy')
y_train = np.load('database/y_train.npy')
y_test = np.load('database/y_test.npy')
y_val = np.load('database/y_val.npy')
```

### Training Final Model
```python
model = CNN_Model()
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['AUC', 'Precision', 'Recall', 'CategoricalAccuracy']
)

history = model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=50,
    batch_size=256,
    callbacks=callbacks
)
```

### Evaluation
```python
y_preds = model.predict(x_test)
y_preds_int = np.argmax(y_preds, axis=1)

evaluator = MultiClassEvaluator(
    y_test_int, y_preds_int, y_preds,
    class_names=['apple', 'avocado', 'banana', 'cherry', 'kiwi', 
                 'mango', 'orange', 'pineapple', 'strawberries', 'watermelon']
)
evaluator.generate_complete_report()
```

---

## 9. Model Saving

Models are saved in multiple formats:
- `model.h5` - Complete model architecture + weights
- `categorical_accuracy_max.keras` - Best model by validation accuracy
- `training.log` - CSV training history
- `logs/` - TensorBoard files

---

## 10. Future Improvements

- Increase training epochs for better convergence
- Address class imbalance (particularly mango and banana)
- Experiment with transfer learning (ResNet, EfficientNet)
- Implement test-time augmentation
- Deploy model via TensorFlow Serving or Flask API

---

## References

- [Fruit Classification Dataset on Kaggle](https://www.kaggle.com/datasets/karimabdulnabi/fruit-classification10-class)
- [Albumentations Documentation](https://albumentations.ai/)
- [Keras Tuner](https://keras.io/keras_tuner/)
- [TensorFlow](https://www.tensorflow.org/)

---

## License

This project is for educational purposes. Please refer to the original dataset license for usage terms.
