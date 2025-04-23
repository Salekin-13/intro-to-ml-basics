import tensorflow as tf

def prepare_dataloader(data, batch_size, resize, is_train=True, is_rgb=False, seed=None):
    X,y = data
    X = tf.cast(X, tf.float32) / 255.0
    y = tf.cast(y, dtype='int32')

    dataset = tf.data.Dataset.from_tensor_slices((X, y))

    def process(X, y):
        
        if is_rgb:
            X = tf.image.rgb_to_grayscale(X)  # Convert 3 channels to 1
        else:
            X = tf.expand_dims(X, axis=-1)     # Add channel dim to grayscale (2D -> 3D)
        
        X = tf.image.resize_with_pad(X, *resize)

        return X, y

    if is_train:
        shuffle_buf = len(X)
        dataset.shuffle(shuffle_buf,seed=seed)

    return dataset.map(process).batch(batch_size).prefetch(tf.data.AUTOTUNE)


class FashionMNIST():
    def __init__(self, batch_size=64, resize=(28,28), seed=None):
        #super().__init__()
        self.batch_size = batch_size
        self.resize = resize
        self.seed = seed

        self.train_ds, self.test_ds = tf.keras.datasets.fashion_mnist.load_data()

    def text_labels(self, indices):
        flabels = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

        return [flabels[int(i)] for i in indices]

    def get_dataloader(self, train):
        data = self.train_ds if train else self.test_ds

        return prepare_dataloader(data, self.batch_size, self.resize, is_train=train, is_rgb=False, seed=self.seed)
    

    

class CIFAR10():
    def __init__(self, batch_size=64, resize=(28,28), seed=None):
        self.batch_size = batch_size
        self.resize = resize
        self.seed = seed

        self.train_ds, self.test_ds = tf.keras.datasets.cifar10.load_data()

    def text_labels(self, indices):
        clabels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

        return [clabels[int(i)] for i in indices]
    
    def get_dataloader(self, train):
        data = self.train_ds if train else self.test_ds

        return prepare_dataloader(data, self.batch_size, self.resize, is_train=train, is_rgb=True, seed=self.seed)
    
    


