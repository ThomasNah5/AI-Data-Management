import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from scipy.stats import mode

# Load the dataset
df = pd.read_csv('Lab_3/penguins.csv')

# Select only the required columns
df = df[['species', 'bill_length_mm', 'bill_depth_mm']]

# Drop rows with missing values
df = df.dropna()

print(f"Dataset shape after cleaning: {df.shape}")
print(f"Species distribution:\n{df['species'].value_counts()}")

# Extract features for clustering
X = df[['bill_length_mm', 'bill_depth_mm']].values

# Encode the species labels for evaluation
le = LabelEncoder()
y_true = le.fit_transform(df['species'])
print(f"\nSpecies labels: {le.classes_}")

# Build K-means clustering model with k=3 (3 penguin species)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
y_pred = kmeans.fit_predict(X)

# Get cluster centroids
centroids = kmeans.cluster_centers_

# Map cluster labels to true labels for accuracy calculation
# Since K-means doesn't know the actual labels, we need to find the best mapping
def map_clusters_to_labels(y_true, y_pred, n_clusters):
    """Map cluster labels to true labels based on majority voting."""
    labels_mapped = np.zeros_like(y_pred)
    for i in range(n_clusters):
        mask = (y_pred == i)
        if np.sum(mask) > 0:
            # Find the most common true label in this cluster
            most_common = mode(y_true[mask], keepdims=True)[0][0]
            labels_mapped[mask] = most_common
    return labels_mapped

y_pred_mapped = map_clusters_to_labels(y_true, y_pred, 3)

# Calculate accuracy
accuracy = accuracy_score(y_true, y_pred_mapped)
print(f"\nK-means Clustering Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

# Visualize the original data distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Original data distribution by species
colors = {'Adelie': 'blue', 'Chinstrap': 'orange', 'Gentoo': 'green'}
for species in df['species'].unique():
    species_data = df[df['species'] == species]
    axes[0].scatter(species_data['bill_length_mm'], species_data['bill_depth_mm'],
                    c=colors.get(species, 'gray'), label=species, alpha=0.6, edgecolors='black')
axes[0].set_xlabel('Bill Length (mm)')
axes[0].set_ylabel('Bill Depth (mm)')
axes[0].set_title('Original Data Distribution by Species')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: K-means clustering results with centroids
scatter = axes[1].scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis', alpha=0.6, edgecolors='black')
axes[1].scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, 
                edgecolors='black', linewidths=2, label='Centroids')
axes[1].set_xlabel('Bill Length (mm)')
axes[1].set_ylabel('Bill Depth (mm)')
axes[1].set_title(f'K-means Clustering Results (Accuracy: {accuracy*100:.2f}%)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('Lab_3/penguin_clustering_results.png', dpi=150)
plt.show()

# Additional: Print centroid locations
print("\nCluster Centroids:")
for i, centroid in enumerate(centroids):
    print(f"  Cluster {i}: Bill Length = {centroid[0]:.2f} mm, Bill Depth = {centroid[1]:.2f} mm")
