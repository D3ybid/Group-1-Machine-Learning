import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

# function to calculate the Euclidean distance between two points
def euclidean_distance(p1, p2):
    return ( (p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2 ) ** 0.5

# k-means clustering algorithm implementation
def kmeans(data, k, max_iters=100):
    centroids = random.sample(data, k)  # randomly pick initial centroids

    for _ in range(max_iters):
        clusters = [[] for _ in range(k)]  # create empty clusters for each centroid
        labels = []

        for point in data:
            distances = [euclidean_distance(point, centroid) for centroid in centroids]
            cluster_index = distances.index(min(distances))  # assign each point to the nearest centroid
            clusters[cluster_index].append(point)
            labels.append(cluster_index)

        new_centroids = []  # calculate new centroids for each cluster
        for cluster in clusters:
            if cluster:
                new_centroid = [
                    sum(point[0] for point in cluster) / len(cluster),
                    sum(point[1] for point in cluster) / len(cluster)
                ]
                new_centroids.append(new_centroid)
            else:
                new_centroids.append(random.sample(data, 1)[0])  # handle empty clusters by picking a random point

        if new_centroids == centroids:  # stop if centroids have stabilized
            break

        centroids = new_centroids

    return centroids, clusters, labels

# load the dataset
file_path = "iris-reduced.csv"
df = pd.read_csv(file_path)

# select two features for clustering
selected_columns = df.columns[:2]
data_points = df[selected_columns].values.tolist()
true_labels = df.iloc[:, -1].tolist()  # Assuming last column contains true labels

# Split dataset into 70% training and 30% testing
random.seed(42)  # For reproducibility
shuffled_indices = list(range(len(data_points)))
random.shuffle(shuffled_indices)

split_index = int(0.7 * len(data_points))
train_indices = shuffled_indices[:split_index]
test_indices = shuffled_indices[split_index:]

train_data = [data_points[i] for i in train_indices]
test_data = [data_points[i] for i in test_indices]
true_test_labels = [true_labels[i] for i in test_indices]

# Apply k-means clustering with k=3 on training data
k = 3
centroids, clusters, train_labels = kmeans(train_data, k)

# Predict test data cluster labels
test_predictions = []
for point in test_data:
    distances = [euclidean_distance(point, centroid) for centroid in centroids]
    test_predictions.append(distances.index(min(distances)))

# Map cluster labels to true labels using majority voting
label_mapping = {}
for cluster_index in range(k):
    cluster_true_labels = [true_labels[j] for j, label in enumerate(train_labels) if label == cluster_index]
    if cluster_true_labels:
        label_mapping[cluster_index] = max(set(cluster_true_labels), key=cluster_true_labels.count)
    else:
        label_mapping[cluster_index] = -1  # Assign unknown if empty cluster

# Convert predicted clusters to corresponding mapped labels
final_predictions = [label_mapping[pred] for pred in test_predictions]

# Compute confusion matrix and F1-score
conf_matrix = confusion_matrix(true_test_labels, final_predictions)
print("Confusion Matrix:")
print(conf_matrix)

print("\nClassification Report:")
print(classification_report(true_test_labels, final_predictions))

# 2D scatter plot for visualization
fig, ax = plt.subplots()
colors = ['r', 'g', 'b']

# Plot training clusters
for i in range(k):
    cluster_points = clusters[i]
    if cluster_points:
        x_values = [point[0] for point in cluster_points]
        y_values = [point[1] for point in cluster_points]
        ax.scatter(x_values, y_values, c=colors[i], label=f'Training Cluster {i+1}', alpha=0.6)

# Plot test data separately
test_x = [point[0] for point in test_data]
test_y = [point[1] for point in test_data]
ax.scatter(test_x, test_y, c='black', marker='x', label="Test Data (30%)", alpha=1)

ax.set_xlabel(selected_columns[0])
ax.set_ylabel(selected_columns[1])
plt.legend()
plt.title("K-Means Clustering: Training & Test Data (30%)")
plt.show()

# Print final centroids and cluster sizes
print("Final Centroids:", centroids)
for i, cluster in enumerate(clusters):
    print(f"Cluster {i+1}: {len(cluster)} points")
