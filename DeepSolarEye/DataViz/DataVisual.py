import matplotlib.pyplot as plt



def visualize_clusters(preprocessed_images, labels, num_clusters):
    fig, axs = plt.subplots(num_clusters, figsize=(10, 20))
    for i in range(num_clusters):
        cluster_indices = np.where(labels == i)[0]
        if len(cluster_indices) > 0:
            # Display the first image of each cluster
            ax = axs[i] if num_clusters > 1 else axs
            ax.imshow(preprocessed_images[cluster_indices[0]])
            ax.set_title(f'Cluster {i}')
            ax.axis('off')
    plt.tight_layout()
    plt.show()
