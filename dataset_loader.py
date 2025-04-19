import tensorflow as tf

def prepare_dataloader(data, batch_size, resize, is_train=True, is_rgb=False):
    def process(X, y):
        X = tf.cast(X, tf.float32) / 255.0
        if is_rgb:
            X = tf.image.rgb_to_grayscale(X)  # Convert 3 channels to 1
        else:
            X = tf.expand_dims(X, axis=3)     # Add channel dim to grayscale (2D -> 3D)
        y = tf.cast(y, dtype='int32')
        return X, y

    resize_fn = lambda X, y: (tf.image.resize_with_pad(X, *resize), y)

    X, y = process(*data)
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    shuffle_buf = len(X) if is_train else 1

    return dataset.shuffle(shuffle_buf).map(resize_fn).batch(batch_size)


class FashionMNIST():
    def __init__(self, batch_size=64, resize=(28,28)):
        #super().__init__()
        self.batch_size = batch_size
        self.resize = resize
        self.train_ds, self.test_ds = tf.keras.datasets.fashion_mnist.load_data()

    def text_labels(self, indices):
        flabels = ['top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

        return [flabels[int(i)] for i in indices]

    def get_dataloader(self, train):
        data = self.train_ds if train else self.test_ds

        return prepare_dataloader(data, self.batch_size, self.resize, is_train=train, is_rgb=False)
    

    

class CIFAR10():
    def __init__(self, batch_size=64, resize=(28,28)):
        self.batch_size = batch_size
        self.resize = resize

        self.train_ds, self.test_ds = tf.keras.datasets.cifar10.load_data()

    def text_labels(self, indices):
        clabels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

        return [clabels[int(i)] for i in indices]
    
    def get_dataloader(self, train):
        data = self.train_ds if train else self.test_ds

        return prepare_dataloader(data, self.batch_size, self.resize, is_train=train, is_rgb=True)
    
    


