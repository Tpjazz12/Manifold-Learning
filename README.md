#  Optimal Transport in Manifold Learning for Image Classification

* This repository examines the efficacy of optimal transport-based manifold learning algorithms versus traditional methods in supervised image classification tasks on datasets like Fashion MNIST, Handwritten MNIST, Dog Breeds Classification, Coil-100 and ImageNet.

## Overview

* This project aims to assess the impact of manifold learning techniques on image classification performance.
  *  The challenge is to compare the effectiveness of traditional and optimal transport-based manifold learning methods in the dimensionality reduction of image datasets for improved classification.
  *  We apply manifold learning algorithms such as PCA, MDS, tSNE, and optimal transport-based Wassmap to transform image datasets, followed by classification using standard machine learning models.
  *  The comparison between manifold learning methods is quantified using visualization quality, stress evaluation, and classification accuracy metrics.

## Summary of Work Done

### Data

* Data:
  * Type:
    * Input: Image datasets (Fashion MNIST, Handwritten MNIST, Dog Breeds Classification and Coil-100), preprocessed for consistent resolution and normalized pixel values.
  * Size: Varied, with the largest being ImageNet at 167.62 GB.
  * Instances: Datasets are split into training, testing, and validation sets in a standard manner. Applying multi embeddings in Manifold Learning: Multi-Dimension-Scaling, Iso-map, t-SNE, Locally Linear and Spectral Embeddings on each of the dataset.

#### Preprocessing / Clean up

* Preprocessing steps included image resizing, grayscale conversion, and pixel normalization.

#### Data Visualization

* Visualizations of manifold embeddings are provided to showcase the data structure and cluster formation.

### Problem Formulation

* Input: High-dimensional image data. Output: Image classifications.
* Models: Comparison of manifold learning models based on traditional and optimal transport theories.
 * MDS
 * ISO-Map
 * T-SNE
 * Locally Linear Embedding
 * Spectral Embedding
 * Wassmap Embeddings  

### Training

* Training details will be provided for replicating results using popular libraries like scikit-learn, TensorFlow, and PyTorch.

### Performance Comparison

* Key metrics include accuracy and error rates. Results will be compared in tabular form and through visualizations.

### Conclusions

* Preliminary findings suggest that optimal transport-based methods may offer better embeddings for image classification.

### Future Work

* Future work may explore deeper integration with neural networks and application to larger datasets.

## How to reproduce results

* Detailed instructions for reproducing the study results and applying the models to new data will be included.

### Overview of files in repository

Each directory in the repository corresponds to a dataset and contains Jupyter notebooks for the manifold learning process and subsequent classification:

*`MDS & ISO map embeddings.ipynb`: Notebooks for applying Multidimensional Scaling and Isomap techniques to datasets.
*`Wassmap.ipynb`: Implementation of the Wasserstein map algorithm for manifold learning.
*`tSNE & LocallyLinearEmbedding & SpectralEmbedding.ipynb`: Notebooks for applying t-SNE, Locally Linear Embedding, and Spectral Embedding algorithms to visualize datasets.

The notebooks contain detailed steps for data processing, manifold learning model application, training classifiers, and evaluating the results. Each notebook is named according to the algorithms implemented within, making it easy to navigate and understand the content's purpose.

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

## Datasets:

* Coil-100: https://www.kaggle.com/datasets/jessicali9530/coil100
* Dog Breeds: https://www.kaggle.com/datasets/mohamedchahed/dog-breeds
* Fashion-MNIST: https://www.kaggle.com/datasets/zalando-research/fashionmnist
* Handwritten-MNIST: https://www.kaggle.com/datasets/dillsunnyb11/digit-recognizer
  
