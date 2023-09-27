import tensorflow as tf
import numpy as np

# Sample data
data = np.random.rand(100, 2).astype(np.float32)

# Number of clusters
num_clusters = 2

# TensorFlow K-Means
kmeans = tf.compat.v1.estimator.experimental.KMeans(
    num_clusters=num_clusters, use_mini_batch=False)

# Input function
def input_fn():
    return tf.compat.v1.train.limit_epochs(
        tf.convert_to_tensor(data, dtype=tf.float32), num_epochs=1)

# Training
kmeans.train(input_fn)

# Get cluster assignments for each data point
cluster_indices = list(kmeans.predict_cluster_index(input_fn))

print("Cluster assignments:", cluster_indices)
