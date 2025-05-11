import pandas as pd
import random
import matplotlib.pyplot as plt
import numpy as np
import os

def save_generation_plots(data_points, generations, selected_columns, out_dir="kmeans_frames"):
    os.makedirs(out_dir, exist_ok=True)
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'orange', 'purple']  # extend if needed
    names = ['vercicolor', 'setosa', 'virginica']

    for gen_index, gen in enumerate(generations):
        fig, ax = plt.subplots()
        labels = gen['labels']
        centroids = gen['centroids']

        for i in range(len(centroids)):
            cluster_points = [p for p, label in zip(data_points, labels) if label == i]
            x_vals = [p[0] for p in cluster_points]
            y_vals = [p[1] for p in cluster_points]
            ax.scatter(x_vals, y_vals, c=colors[i % len(colors)], label=f'cluseter {i}')

        # Plot centroids
        cx = [c[0] for c in centroids]
        cy = [c[1] for c in centroids]
        ax.scatter(cx, cy, c='black', marker='X', s=100, label='Centroids')

        ax.set_xlabel(selected_columns[0])
        ax.set_ylabel(selected_columns[1])
        ax.set_title(f"K-Means - Generation {gen_index}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{out_dir}/gen_{gen_index:03d}.png")
        plt.close()


# function to calculate the Euclidean distance between two points
def euclidean_distance(p1, p2):
    return ( (p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2 ) ** 0.5

def kmeans(data, k, max_iters=100):
    centroids = random.sample(data, k)  # Randomly pick initial centroids
    generations = []  # store states at each iteration
    this_generation = 0
    for _ in range(max_iters):
        clusters = [[] for _ in range(k)]
        labels = []
        this_generation += 1
        for point in data:
            distances = [euclidean_distance(point, centroid) for centroid in centroids]
            cluster_index = np.argmin(distances)  # Assign to closest centroid
            clusters[cluster_index].append(point)
            labels.append(cluster_index)

        # Save the current generation
        generations.append({
            'centroids': [c.copy() for c in centroids],
            'labels': labels.copy()
        })

        print(f"GENERATION {this_generation}")
        print(f"CENTROIDS: {centroids}")
        for i, cluster in enumerate(clusters):
            print(f"Cluster {i+1}: {len(cluster)} points")

        new_centroids = []
        for cluster in clusters:
            if cluster:
                new_centroid = np.mean(cluster, axis=0).tolist()
                new_centroids.append(new_centroid)
            else:
                new_centroids.append(random.choice(data))  # handle empty clusters

        if np.allclose(new_centroids, centroids, atol=1e-6):
            break

        centroids = new_centroids

    return centroids, clusters, labels, this_generation, generations

# # K-Means Algorithm
# def kmeans(data, k, max_iters=100):
#     centroids = random.sample(data, k)  # Randomly pick initial centroids

#     for _ in range(max_iters):
#         clusters = [[] for _ in range(k)]
#         labels = []

#         for point in data:
#             distances = [euclidean_distance(point, centroid) for centroid in centroids]
#             cluster_index = np.argmin(distances)  # Assign to closest centroid
#             clusters[cluster_index].append(point)
#             labels.append(cluster_index)

#         new_centroids = []
#         for cluster in clusters:
#             if cluster:
#                 new_centroid = np.mean(cluster, axis=0).tolist()
#                 new_centroids.append(new_centroid)
#             else:
#                 new_centroids.append(random.choice(data))  # Handle empty clusters

#         if np.allclose(new_centroids, centroids, atol=1e-6):
#             break

#         centroids = new_centroids

#     return centroids, clusters, labels, total_gens



# load the dataset
file_path = "data/iris-reduced.csv"
df = pd.read_csv(file_path)

# df = df.sample(n=100, random_state=42)  # sample 100 random rows to work with

# select two features for clustering (changed from 3 to 2 features)
selected_columns = df.columns[:2]
data_points = df[selected_columns].values.tolist()

# # save the reduced dataset to a new CSV file
# df[selected_columns].to_csv("wine-clustering-reduced.csv", index=False)

# apply k-means clustering with k=3
k = 3
centroids, clusters, labels, total_gens, generations = kmeans(data_points, k)
save_generation_plots(data_points, generations, selected_columns)



# 2D scatter plot for visualization
fig, ax = plt.subplots()  # create a 2D plot
colors = ['r', 'g', 'b'] #versicolor, setosa, virginica
names = ['vercicolor', 'setosa', 'virginica']


for i in range(k):
    cluster_points = clusters[i]
    if cluster_points:
        x_values = [point[0] for point in cluster_points]
        y_values = [point[1] for point in cluster_points]
        ax.scatter(x_values, y_values, c=colors[i], label=f'cluster{i}')


# Plot centroids
centroid_x = [c[0] for c in centroids]
centroid_y = [c[1] for c in centroids]
ax.scatter(centroid_x, centroid_y, c='purple', marker='X', s=100, label='Centroids')


ax.set_xlabel(selected_columns[0])  # set label for the x-axis
ax.set_ylabel(selected_columns[1])  # set label for the y-axis
plt.legend()  # show legend to identify clusters
plt.title("K-Means Clustering Visualization (2D)")  # plot title
plt.show()

# print the final centroids and the size of each cluster
print("Final Centroids:", centroids)
for i, cluster in enumerate(clusters):
    print(f"Cluster {i+1}: {len(cluster)} points")

print(f"TOTAL NUMBER OF GENERATIONS: {total_gens}")
