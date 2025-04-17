import tensorflow as tf

class FashionMNIST():
    def __init__(self, batch_size=64, resize=(28,28)):
        #super().__init__()
        self.batch_size = batch_size
        self.resize = resize
        self.train_ds, self.test_ds = tf.keras.datasets.fashion_mnist.load_data()

    def text_labels(self, indices):
        labels = ['top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

        return [labels[int(i)] for i in indices]

    def get_dataloader(self, train):
        data = self.train_ds if train else self.test_ds

        process = lambda X,y: (tf.expand_dims(tf.cast(X, tf.float32), axis=3)/255.0,
                               tf.cast(y, dtype='int32'))
        
        resize_fn = lambda X,y: (tf.image.resize_with_pad(X, *self.resize), y)

        X,y = process(*data)
        dataset = tf.data.Dataset.from_tensor_slices((X,y))
        shuffle_buf = len(X) if train else 1

        return dataset.shuffle(shuffle_buf).map(resize_fn).batch(self.batch_size)
    

class CIFAR10():
    def __init__(self, batch_size=64, resize=(28,28)):
        self.batch_size = batch_size
        self.resize = resize

        self.train_ds, self.test_ds = tf.keras.datasets.cifar10.load_data()

    def text_labels(self, indices):
        labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

        return [labels[int(i)] for i in indices]
    
    def get_dataloader(self, train):
        data = self.train_ds if train else self.test_ds

        process = lambda X,y: (tf.image.rgb_to_grayscale(tf.cast(X, tf.float32))/255.0,
                               tf.cast(y, dtype='int32'))
        
        resize_fn = lambda X,y: (tf.image.resize_with_pad(X, *self.resize), y)

        X,y = process(*data)
        dataset = tf.data.Dataset.from_tensor_slices((X,y))
        shuffle_buf = len(X) if train else 1

        return dataset.shuffle(shuffle_buf).map(resize_fn).batch(self.batch_size)
    