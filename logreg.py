import tensorflow as tf
import numpy as np
# Sample data
X = np.random.rand(100, 2).astype(np.float32)
y = (X[:, 0] + X[:, 1] > 1).astype(np.int32)

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[2], activation='sigmoid')
])

# Compile the model
model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X, y, epochs=100)
