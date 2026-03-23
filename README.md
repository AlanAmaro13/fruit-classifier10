# Fruit Classification Dataset - Data Acquisition & Preprocessing

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/)

A comprehensive data preprocessing pipeline for the Fruit Classification (10-Class) dataset from Kaggle. This notebook handles dataset acquisition, exploratory data analysis, image resizing, normalization, and stratified dataset splitting.

## 📋 Overview

This preprocessing pipeline prepares the Fruit Classification dataset for machine learning models by:
- Downloading the dataset from Kaggle Hub
- Analyzing dataset structure and file formats
- Examining image resolution distribution
- Resizing all images to a standardized size (128x128)
- Normalizing pixel values
- Performing stratified train/validation/test splits
- Saving processed data for model training

## 📊 Dataset Information

The dataset contains images of 10 different fruit categories:
- Apple
- Banana
- Cherry
- Kiwi
- Orange
- Peach
- Pineapple
- Strawberry
- Watermelon
- (Other fruits as present in the dataset)

### Original Dataset Statistics
- **Total images**: 3,374 images
- **Format**: JPEG
- **Original resolutions**: Multiple resolutions (most common: 275x183)
- **Train/Test structure**: Original dataset comes with train/test splits

## 🔧 Requirements

```bash
pip install kagglehub matplotlib seaborn numpy opencv-python scikit-learn tqdm
```

## 📁 Directory Structure

```
fruit_classification/
├── database/
│   ├── x_train.npy
│   ├── x_test.npy
│   ├── x_val.npy
│   ├── y_train.npy
│   ├── y_test.npy
│   └── y_val.npy
└── preprocessing_notebook.ipynb
```

## 🚀 Key Features

### 1. **Data Acquisition**
- Automatic download from Kaggle Hub
- Directory tree visualization
- File format analysis

### 2. **Exploratory Data Analysis**
- Resolution distribution analysis with visualization
- Class distribution analysis
- File extension verification
- Sample image visualization

### 3. **Image Preprocessing**
- Standardized resizing to 128x128 pixels
- Normalization to [0, 1] range
- Color space conversion (RGB/BGR handling)
- Batch processing with progress bars

### 4. **Data Splitting**
- Stratified splitting to maintain class distribution
- Configurable train/validation/test ratios (default: 70/15/15)
- Comprehensive split statistics and visualizations
- Class balance verification across splits

### 5. **Data Export**
- Numpy array format for efficient loading
- Separate storage for features (X) and labels (y)
- Organized directory structure for easy access

## 📈 Visualizations

The notebook generates several informative visualizations:
- **Resolution distribution** - Horizontal bar chart showing image dimension frequencies
- **Class distribution** - Distribution of samples across fruit categories
- **Split distribution** - Stacked bar chart showing class distribution across train/val/test sets
- **Sample images** - Visual inspection of processed images

## 🔄 Processing Pipeline

1. **Data Loading**
   ```
   Kaggle Hub Download → File Path Collection → Extension Analysis
   ```

2. **Image Processing**
   ```
   Image Reading → Resolution Analysis → Resizing → Normalization
   ```

3. **Data Preparation**
   ```
   Label Extraction → Class Correction → Stratified Split → Export
   ```

## 💻 Usage

```python
# After running the notebook, load the preprocessed data:
import numpy as np

X_train = np.load('database/x_train.npy')
X_test = np.load('database/x_test.npy')
X_val = np.load('database/x_val.npy')
y_train = np.load('database/y_train.npy')
y_test = np.load('database/y_test.npy')
y_val = np.load('database/y_val.npy')

print(f"Training set: {X_train.shape} images")
print(f"Validation set: {X_val.shape} images")
print(f"Test set: {X_test.shape} images")
```

## 📊 Output Data Format

- **X_train, X_val, X_test**: 4D numpy arrays of shape (n_samples, 128, 128, 3)
  - Pixel values normalized to range [0, 1]
  - RGB color channels

- **y_train, y_val, y_test**: 1D numpy arrays of shape (n_samples,)
  - String labels for fruit categories
  - Balanced distribution across classes

## 🎯 Benefits of This Preprocessing

- **Standardized Input Size**: All images resized to 128×128 pixels for consistent model input
- **Normalized Data**: Pixel values scaled to [0, 1] for better model convergence
- **Stratified Splitting**: Maintains class distribution across all splits
- **Balanced Dataset**: Ensures each fruit category has representative samples in all sets
- **Reproducibility**: Fixed random seed ensures consistent splits

## 📝 Notes

- The preprocessing automatically handles label inconsistencies (e.g., "bananas" vs "banana", "stawberries" vs "strawberries")
- The "preds" folder is excluded from training as it contains unlabeled data
- Image resizing uses OpenCV's INTER_AREA interpolation for high-quality downscaling
- The script saves all processed data in efficient numpy binary format (.npy)

## 🔗 References

- [Fruit Classification (10-Class) Dataset on Kaggle](https://www.kaggle.com/datasets/karimabdulnabi/fruit-classification10-class)
- [OpenCV Documentation](https://docs.opencv.org/)
- [Scikit-learn Train-Test Split Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)

## 👥 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or feedback, please open an issue on GitHub.


