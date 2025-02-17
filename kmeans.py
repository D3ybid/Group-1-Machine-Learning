import pandas as pd
import random
import matplotlib.pyplot as plt

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
                # Using the formula (x', y') = [(x1 + x2 + x3) / n, (y1 + y2 + y3) / n]
                new_centroid = [
                    sum(point[0] for point in cluster) / len(cluster),  # x' = (x1 + x2 + x3) / n
                    sum(point[1] for point in cluster) / len(cluster)   # y' = (y1 + y2 + y3) / n
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

# df = df.sample(n=100, random_state=42)  # sample 100 random rows to work with

# select two features for clustering (changed from 3 to 2 features)
selected_columns = df.columns[:2]
data_points = df[selected_columns].values.tolist()

# # save the reduced dataset to a new CSV file
# df[selected_columns].to_csv("wine-clustering-reduced.csv", index=False)

# apply k-means clustering with k=3
k = 3
centroids, clusters, labels = kmeans(data_points, k)

# 2D scatter plot for visualization
fig, ax = plt.subplots()  # create a 2D plot
colors = ['r', 'g', 'b'] #versicolor, setosa, virginica

for i in range(k):
    cluster_points = clusters[i]
    if cluster_points:
        x_values = [point[0] for point in cluster_points]
        y_values = [point[1] for point in cluster_points]
        ax.scatter(x_values, y_values, c=colors[i], label=f'Cluster {i+1}')

ax.set_xlabel(selected_columns[0])  # set label for the x-axis
ax.set_ylabel(selected_columns[1])  # set label for the y-axis
plt.legend()  # show legend to identify clusters
plt.title("K-Means Clustering Visualization (2D)")  # plot title
plt.show()

# print the final centroids and the size of each cluster
print("Final Centroids:", centroids)
for i, cluster in enumerate(clusters):
    print(f"Cluster {i+1}: {len(cluster)} points")
