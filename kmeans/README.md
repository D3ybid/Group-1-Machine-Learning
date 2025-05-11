# Group-1-Machine-Learning - K-Means Clustering on the Iris Dataset

# Overview

This repository contains Python scripts for performing K-Means clustering on the Iris dataset. The implementation includes data preprocessing, determining the optimal number of clusters using the Elbow Method, and visualizing the clustered data.

# Files

**dataprep.py**: Prepares the dataset by selecting relevant features and reducing the dataset to 100 rows.

**elbow.py**: Implements the Elbow Method to determine the optimal number of clusters for K-Means.

**kmeans.py**: Performs K-Means clustering on the processed dataset and visualizes the results.

**testing.py**: Evaluates K-Means clustering using accuracy metrics and confusion matrix.

# How to Run

## I. Repository

### 1. Clone the repository on your machine

```bash
git clone https://github.com/D3ybid/Group-1-Machine-Learning.git
```

### 2. Change the directory to the Kmeans implementation

```bash
cd kmeans
```

## II. Dependencies

Ensure you have the following Python libraries installed before running the scripts:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## III. Usage

### 1. Data Preparation

Run dataprep.py to preprocess the dataset:

```bash
python src/dataprep.py
```

This script will generate iris-reduced.csv containing 100 samples with selected features.

### 2. Determine Optimal Clusters (Elbow Method)

Run elbow.py to visualize the Elbow Method and determine the best value for k:

```bash
python src/elbow.py
```

A plot will be isplayed to help identify the optimal number of clusters.

### 3. Perform K-Means Clustering

Run kmeans.py to apply K-Means clustering to the dataset and visualize the results:

```bash
python src/kmeans.py
```

A scatter plot will be generated, showing the clustered data points.

### 4. Evaluate Model Performance

Run test2.py to assess clustering performance:

```bash
python src/testing.py
```

This script computes accuracy, confusion matrix, and classification reports for model evaluation.

## Output

**iris-reduced.csv**: Processed dataset with selected features.

Visualization plots for the Elbow Method and K-Means clustering.

Cluster centroids and performance evaluation metrics printed to the console.

# License

This project is open-source and available for use and modification.

# Authors

Agapito, Hazel Anne A.

Agno, Tricia Mae C.

Andino, Coren Andrei

Aromin, Oliver Luis S.

Balingit, Jose Emmanuel T.

Bautista, Danlor

Borja, Jahara Kate B.

Buenaventura, John David C.
