# Deep Learning Project 1: CNN Image Classification with CINIC-10

## Project Overview

This project focuses on implementing and evaluating convolutional neural networks (CNNs) for image classification tasks using the CINIC-10 dataset. The project involves comprehensive experimentation with various hyperparameters, data augmentation techniques, and few-shot learning methods to understand their impact on model performance.

## Dataset

### CINIC-10 Dataset
- **Description**: The CINIC-10 dataset is a large-scale image classification dataset containing 10 classes:
  - Airplane
  - Automobile
  - Bird
  - Cat
  - Deer
  - Dog
  - Frog
  - Horse
  - Ship
  - Truck

- **Size**: Approximately 100,000 images (training: ~50,000, validation: ~10,000, test: ~40,000)
- **Resolution**: Images are 32x32 pixels in RGB format
- **Source**: Available at https://www.kaggle.com/datasets/mengcius/cinic10

## Project Objectives

### Core Requirements
1. Investigate influence of hyperparameters on model performance:
   - At least 2 training-related hyperparameters
   - At least 2 regularization-related hyperparameters

2. Evaluate data augmentation techniques:
   - At least 3 standard operations
   - At least 1 advanced technique (cutmix, cutout, AutoAugment)

3. Implement few-shot learning method

4. Compare performance with reduced training sets

5. Optional: Implement ensemble methods

## Implementation Plan

### Phase 1: Environment Setup and Data Preparation
- Set up Python environment with required libraries
- Download and preprocess CINIC-10 dataset
- Establish reproducibility with fixed random seeds

### Phase 2: Baseline Model Development
- Implement CNN architecture for image classification
- Establish baseline performance metrics
- Validate data loading and preprocessing pipeline

### Phase 3: Hyperparameter Analysis
- Experiment with different learning rates
- Test various batch sizes
- Evaluate dropout rates and regularization strengths
- Analyze impact of optimizer choices

### Phase 4: Data Augmentation Studies
- Implement standard augmentation techniques:
  - Random horizontal flip
  - Random crop and resize
  - Color jittering
- Implement advanced techniques:
  - Cutout augmentation
- Compare performance with and without augmentation

### Phase 5: Few-Shot Learning Implementation
- Implement one few-shot learning method:
  - Siamese networks for few-shot classification
  - Or support vector machine with pre-trained features

### Phase 6: Reduced Dataset Analysis
- Evaluate model performance with 10%, 25%, 50% of training data
- Analyze performance degradation trends

### Phase 7: Ensemble Methods (Optional)
- Implement voting or stacking ensemble techniques
- Evaluate ensemble performance improvement

### Phase 8: Comprehensive Evaluation and Reporting
- Statistical analysis of results
- Performance visualization and comparison
- Documentation of findings and conclusions

## Libraries and Tools Used

### Primary Libraries:
- `tensorflow` or `pytorch`: Deep learning framework
- `numpy`, `pandas`: Data manipulation and analysis
- `matplotlib`, `seaborn`: Visualization tools
- `scikit-learn`: Machine learning utilities and evaluation metrics

### Data Processing:
- `PIL` or `OpenCV`: Image processing operations
- `Keras` (if using TensorFlow): High-level neural network API

## Project Structure

```
Project 1/
├── data/
│   ├── cinic-10/
│   │   ├── train/
│   │   ├── validation/
│   │   └── test/
├── models/
│   ├── baseline_cnn.h5
│   ├── augmented_model.h5
│   └── few_shot_model.h5
├── results/
│   ├── metrics.csv
│   └── visualizations/
├── src/
│   ├── data_preprocessing.py
│   ├── model_architecture.py
│   ├── hyperparameter_analysis.py
│   ├── augmentation_studies.py
│   ├── few_shot_learning.py
│   └── evaluation.py
├── notebooks/
│   ├── baseline_experiment.ipynb
│   ├── hyperparameter_tuning.ipynb
│   └── augmentation_analysis.ipynb
├── README.md
└── requirements.txt
```

## Expected Deliverables

1. **Comprehensive Report** (as per project guidelines):
   - Abstract
   - Research problem description
   - Theoretical introduction and literature review
   - Experiment methodology
   - Statistically processed results
   - Conclusions and future work

2. **Code Implementation**:
   - Reproducible experiments
   - Well-documented code with proper structure
   - Clear parameter configurations

3. **Presentation**:
   - 10-minute slideshow summarizing key findings
   - Focus on methodology, results, and implications

## Reproducibility Measures

To ensure reproducible results:
- Fixed random seeds for all operations (numpy, tensorflow, python)
- Version-controlled dependencies in requirements.txt
- Detailed experiment logging and configuration files

## Performance Metrics

- Accuracy (overall classification accuracy)
- Top-1 and Top-5 error rates
- Confusion matrices for each class
- Training/validation loss curves
- Computational efficiency measures

## Assessment Criteria

1. **Research Process**: Methodology, experimental design, and thorough investigation
2. **Report Quality**: Formal documentation with proper structure and content
3. **Novelty/Originality**: Creative approaches to solving problems or unique insights
4. **Model Performance**: Statistical significance and robustness of results

## Resource Considerations

Due to computing power limitations:
- Implement smaller models when necessary
- Use data augmentation instead of larger datasets
- Consider reduced training set sizes for initial experiments
- Optimize hyperparameters to find efficient solutions

## References and Resources

- CINIC-10 dataset documentation
- Keras ImageDataGenerator for data augmentation
- PyTorch torchvision transforms for image processing
- State-of-the-art CNN architectures and techniques

This project aims to provide comprehensive understanding of CNN behavior under various conditions, with practical insights applicable to real-world image classification problems.
```
