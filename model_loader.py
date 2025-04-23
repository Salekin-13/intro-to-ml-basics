import tensorflow as tf
from sklearn.linear_model import LogisticRegressionCV
from tensorflow.keras import regularizers
import numpy as np

def set_seed(seed_value):
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)


def split_large_dataset(dataset, val_ratio=0.2, batch_size=64, total_size=None):
    """
    Splits a batched and shuffled tf.data.Dataset into train and validation sets using a streaming-safe approach.

    Args:
        dataset: tf.data.Dataset (shuffled, batched)
        val_ratio: Fraction for validation set
        batch_size: Batch size after re-batching
        total_size: Total number of samples (required if not inferable)

    Returns:
        train_ds, val_ds: The split datasets
    """
    # Step 1: Unbatch to access individual samples
    dataset = dataset.unbatch().enumerate()

    # Step 2: Estimate or use provided total size
    if total_size is None:
        raise ValueError("For large datasets, please provide total_size to avoid loading everything into memory.")

    val_start = int(total_size * (1 - val_ratio))


    # Step 3: Create train and val datasets by filtering on index
    train_ds = (
        dataset
        .filter(lambda i, data: i < val_start)
        .map(lambda i, data: data)
        .batch(batch_size)
    )

    val_ds = (
        dataset
        .filter(lambda i, data: i >= val_start)
        .map(lambda i, data: data)
        .batch(batch_size)
    )

    return train_ds, val_ds


class LogReg():
    def __init__(self, input_shape, num_classes, seed_value= None):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.sk_model = None
        self.tf_model = None

        if seed_value is not None:
            set_seed(seed_value)
            self.seed_value = seed_value

    def skLogReg(self):
        self.sk_model = LogisticRegressionCV(
            Cs=10,
            cv=5,
            multi_class='multinomial',
            max_iter= 1000,
            solver='lbfgs',
            penalty='l2'
        )

    def tfLogReg(self, reg_type=None, reg_strength=1e-4, alpha=1e-5, beta=1e-3, learning_rate=0.001, n_epochs=5):
        tf.keras.backend.clear_session()

        if reg_type == 'l1':
            reg = regularizers.l1(reg_strength)
        elif reg_type == 'l2':
            reg = regularizers.l2(reg_strength)
        elif reg_type == 'elastic':
            reg = regularizers.l1_l2(l1=alpha, l2=beta)
        else:
            reg=None

        self.tf_model = tf.keras.Sequential([
            tf.keras.Input(shape=self.input_shape),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(self.num_classes, 
                                  activation= 'softmax',
                                  kernel_regularizer=reg,
                                  kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.seed_value)  # Seed for weight initialization
                                  )
        ])

        self.tf_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )