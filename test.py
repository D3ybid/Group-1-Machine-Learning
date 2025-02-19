import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Function to compute Euclidean distance
def euclidean_distance(p1, p2):
    return np.sqrt(np.sum((np.array(p1) - np.array(p2)) ** 2))

# K-Means++ Initialization
def kmeans_plus_plus_init(data, k):
    centroids = [random.choice(data)]
    for _ in range(1, k):
        distances = [min([euclidean_distance(point, c) for c in centroids]) for point in data]
        probabilities = distances / np.sum(distances)
        cumulative_probabilities = np.cumsum(probabilities)
        r = random.random()
        for i, p in enumerate(cumulative_probabilities):
            if r < p:
                centroids.append(data[i])
                break
    return centroids

# K-Means Clustering Algorithm
def kmeans(data, k, max_iters=100):
    centroids = kmeans_plus_plus_init(data, k)  # Use K-Means++
    for _ in range(max_iters):
        clusters = [[] for _ in range(k)]
        labels = []
        for point in data:
            distances = [euclidean_distance(point, centroid) for centroid in centroids]
            cluster_index = np.argmin(distances)
            clusters[cluster_index].append(point)
            labels.append(cluster_index)
        
        new_centroids = []
        for cluster in clusters:
            if cluster:
                new_centroid = np.mean(cluster, axis=0).tolist()
                new_centroids.append(new_centroid)
            else:
                new_centroids.append(random.choice(data))  # Handle empty clusters
        
        if np.allclose(new_centroids, centroids, atol=1e-6):
            break
        centroids = new_centroids
    
    return centroids, clusters, labels

# Load dataset
file_path = "iris-reduced.csv"
df = pd.read_csv(file_path)
selected_columns = df.columns[:2]
data_points = df[selected_columns].values.tolist()
true_labels = df.iloc[:, -1].tolist()  # Assuming last column contains true labels

# Train-Test Split
random.seed(42)
indices = list(range(len(data_points)))
random.shuffle(indices)
split_idx = int(0.7 * len(data_points))
train_indices, test_indices = indices[:split_idx], indices[split_idx:]
train_data = [data_points[i] for i in train_indices]
test_data = [data_points[i] for i in test_indices]
true_test_labels = [true_labels[i] for i in test_indices]

# Apply K-Means with k=3
k = 3
centroids, clusters, train_labels = kmeans(train_data, k)

# Predict Test Data Cluster Labels
test_predictions = [np.argmin([euclidean_distance(point, c) for c in centroids]) for point in test_data]

# Map Cluster Labels to True Labels (Majority Voting)
label_mapping = {}
for cluster_idx in range(k):
    cluster_true_labels = [true_labels[i] for i, label in zip(train_indices, train_labels) if label == cluster_idx]
    label_mapping[cluster_idx] = max(set(cluster_true_labels), key=cluster_true_labels.count) if cluster_true_labels else -1

# Convert Predicted Clusters to Mapped Labels
final_predictions = [label_mapping[pred] for pred in test_predictions]

# Handle Unassigned Labels
unique_true_labels = set(true_labels)
unmapped_labels = unique_true_labels - set(label_mapping.values())
if unmapped_labels:
    for cluster_idx in range(k):
        if label_mapping[cluster_idx] == -1:
            label_mapping[cluster_idx] = random.choice(list(unmapped_labels))
            unmapped_labels.remove(label_mapping[cluster_idx])
            if not unmapped_labels:
                break

# Compute Metrics
conf_matrix = confusion_matrix(true_test_labels, final_predictions)
print("Confusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(classification_report(true_test_labels, final_predictions, zero_division=1))

accuracy = accuracy_score(true_test_labels, final_predictions)
print(f"\nOverall Accuracy: {accuracy:.2f}")

# Plot Clusters
fig, ax = plt.subplots()
colors = ['r', 'g', 'b']
for i in range(k):
    cluster_points = clusters[i]
    if cluster_points:
        x_vals, y_vals = zip(*cluster_points)
        ax.scatter(x_vals, y_vals, c=colors[i], label=f'Training Cluster {i+1}', alpha=0.6)

# Plot Test Data
test_x, test_y = zip(*test_data)
ax.scatter(test_x, test_y, c='black', marker='x', label="Test Data (30%)", alpha=1)
ax.set_xlabel(selected_columns[0])
ax.set_ylabel(selected_columns[1])
plt.legend()
plt.title("Optimized K-Means Clustering")
plt.show()

# Print Final Centroids & Cluster Sizes
print("Final Centroids:", centroids)
for i, cluster in enumerate(clusters):
    print(f"Cluster {i+1}: {len(cluster)} points")