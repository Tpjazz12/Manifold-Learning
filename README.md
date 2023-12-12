# Project Title: Optimal Transport in Manifold Learning for Image Classification

* This repository examines the efficacy of optimal transport-based manifold learning algorithms versus traditional methods in supervised image classification tasks on datasets like MNIST and Fashion MNIST.

## Overview

* This project aims to assess the impact of manifold learning techniques on image classification performance.
  * **Definition of the tasks / challenge**: The challenge is to compare the effectiveness of traditional and optimal transport-based manifold learning methods in the dimensionality reduction of image datasets for improved classification.
  * **Your approach**: We apply manifold learning algorithms such as PCA, MDS, tSNE, and optimal transport-based Wassmap to transform image datasets, followed by classification using standard machine learning models.
  * **Summary of the performance achieved**: The comparison between manifold learning methods is quantified using visualization quality, stress evaluation, and classification accuracy metrics.

## Summary of Work Done

### Data

* Data:
  * Type:
    * Input: Image datasets (MNIST, Fashion MNIST, etc.), preprocessed for consistent resolution and normalized pixel values.
  * Size: Varied, with the largest being ImageNet at 167.62 GB.
  * Instances: Datasets are split into training, testing, and validation sets in a standard manner.

#### Preprocessing / Clean up

* Preprocessing steps included image resizing, grayscale conversion, and pixel normalization.

#### Data Visualization

* Visualizations of manifold embeddings are provided to showcase the data structure and cluster formation.

### Problem Formulation

* Input: High-dimensional image data. Output: Image classifications.
* Models: Comparison of manifold learning models based on traditional and optimal transport theories.

### Training

* Training details will be provided for replicating results using popular libraries like scikit-learn, TensorFlow, and PyTorch.

### Performance Comparison

* Key metrics include accuracy and error rates. Results will be compared in tabular form and through visualizations such as ROC curves.

### Conclusions

* Preliminary findings suggest that optimal transport-based methods may offer better embeddings for image classification.

### Future Work

* Future work may explore deeper integration with neural networks and application to larger datasets.

## How to reproduce results

* Detailed instructions for reproducing the study results and applying the models to new data will be included.

### Overview of files in repository

* `utils.py`: Helper functions for data processing.
* `preprocess.ipynb`: Notebook for initial data preparation.
* `models.py`: Definitions for manifold learning models.
* `train.ipynb`: Notebook for model training.
* `evaluate.ipynb`: Notebook for model evaluation and comparison.

### Software Setup

* Required packages include scikit-learn, TensorFlow, PyTorch, and Python Optimal Transport.

### Data

* Data sources will be provided with instructions for downloading and preprocessing.

### Training

* Step-by-step guidance will be provided for model training.

#### Performance Evaluation

* Instructions for evaluating model performance will be detailed.

## Citations

* References to the foundational literature, including works by Peyré & Cuturi (2019) and Gonzalez-Castillo (2023), will be cited.
