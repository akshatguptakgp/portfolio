import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
keypoints_list = keypoints_array
labels_list = labels

keypoints_array = np.array(keypoints_list)

# Perform t-SNE
tsne = TSNE(n_components=2, random_state=42)
keypoints_tsne = tsne.fit_transform(keypoints_array)

# Create a color mapping for each unique label
unique_labels = np.unique(labels_list)
color_map = {label: idx for idx, label in enumerate(unique_labels)}
colors = [color_map[label] for label in labels_list]

# Create a scatter plot
plt.figure(figsize=(10, 8))
scatter = plt.scatter(keypoints_tsne[:, 0], keypoints_tsne[:, 1], c=colors, cmap='viridis', alpha=0.6)

# Create a legend
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=scatter.cmap(scatter.norm(color_map[label])), markersize=10) for label in unique_labels]
plt.legend(handles, unique_labels, title="Labels")

plt.title('t-SNE Plot of Pose Keypoints')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.grid()
plt.show()