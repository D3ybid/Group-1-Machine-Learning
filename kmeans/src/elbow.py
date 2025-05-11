import pandas as pd
import random
import matplotlib.pyplot as plt

# Function to calculate the Euclidean distance
def euclidean_distance(p1, p2):
    return sum((p2[i] - p1[i]) ** 2 for i in range(len(p1)))

# K-means clustering algorithm
def kmeans(data, k, max_iters=100):
    centroids = random.sample(data, k)  # Randomly select initial centroids

    for _ in range(max_iters):
        clusters = [[] for _ in range(k)]  # Create empty clusters
        labels = []

        for point in data:
            distances = [euclidean_distance(point, centroid) for centroid in centroids]
            cluster_index = distances.index(min(distances))  # Assign to the nearest centroid
            clusters[cluster_index].append(point)
            labels.append(cluster_index)

        new_centroids = []
        for cluster in clusters:
            if cluster:
                new_centroid = [
                    sum(point[0] for point in cluster) / len(cluster),  # Mean X
                    sum(point[1] for point in cluster) / len(cluster)   # Mean Y
                ]
                new_centroids.append(new_centroid)
            else:
                new_centroids.append(random.sample(data, 1)[0])  # Handle empty clusters

        if new_centroids == centroids:  # Stop if centroids stabilize
            break

        centroids = new_centroids

    return centroids, clusters, labels

# Load the dataset
file_path = "data/iris-reduced.csv"
df = pd.read_csv(file_path)
df = df.sample(n=100, random_state=42)  # Sample 100 random rows

# Select two features for clustering
selected_columns = df.columns[:2]
data_points = df[selected_columns].values.tolist()

# Compute WCSS for k = 1 to 10
wcss_values = []
k_values = range(1, 11)

for k in k_values:
    centroids, clusters, _ = kmeans(data_points, k)

    # Compute WCSS
    wcss = 0
    for i in range(k):
        centroid = centroids[i]
        for point in clusters[i]:
            wcss += euclidean_distance(point, centroid)  # Sum of squared distances

    wcss_values.append(wcss)

# Plot the Elbow Method
plt.figure(figsize=(8, 5))
plt.plot(k_values, wcss_values, marker='o', linestyle='-', color='b')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('WCSS')
plt.title('Elbow Method for Optimal k')
plt.xticks(k_values)
plt.grid(True)
plt.show()
