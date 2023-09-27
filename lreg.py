import tensorflow as tf
import numpy as np

# Sample data
X = np.random.rand(100).astype(np.float32)
y = 2 * X + 1 + np.random.randn(100).astype(np.float32) * 0.5

# Defining model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1])
])

# Compile the model
model.compile(optimizer='sgd', loss='mean_squared_error')

# Train the model
model.fit(X, y, epochs=100)
