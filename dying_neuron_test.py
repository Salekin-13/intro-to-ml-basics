import numpy as np
import tensorflow as tf

# Callback to monitor dying ReLU neurons after each epoch without needing sub-model
class DyingNeuronMonitor(tf.keras.callbacks.Callback):
    def __init__(self, data, threshold=0.99):
        super().__init__()
        self.data = data
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs=None):
        # Grab a single batch from the validation data
        X_batch, _ = next(iter(self.data))
        # Ensure tensor
        x = tf.convert_to_tensor(X_batch)

        # Identify ReLU activation layers (hidden ones)
        relu_layers = [layer for layer in self.model.layers
                       if isinstance(layer, tf.keras.layers.Activation)
                       and layer.activation == tf.keras.activations.relu]

        activations = []
        # Forward pass through layers, collecting after each ReLU
        for layer in self.model.layers:
            # Some layers (like Dropout/BatchNorm) accept 'training' arg
            try:
                x = layer(x, training=False)
            except TypeError:
                x = layer(x)
            if layer in relu_layers:
                activations.append(x.numpy())

        # Compute and report dying neurons per hidden layer
        reports = []
        for idx, act in enumerate(activations, start=1):
            zero_frac = np.mean(act == 0, axis=0)
            dying = np.sum(zero_frac > self.threshold)
            total = act.shape[1]
            reports.append(f"L{idx}: {dying}/{total}")
        print(f"[Epoch {epoch+1}] Dying neurons -> " + "; ".join(reports))