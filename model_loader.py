import tensorflow as tf
from sklearn.linear_model import LogisticRegressionCV
from tensorflow.keras import regularizers
import numpy as np

def set_seed(seed_value):
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)


class LogReg():
    """
        initializes a Logistic Regression model
    """
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

    def tfLogReg(self, reg_type=None, reg_strength=1e-4, alpha=1e-5, beta=1e-3, learning_rate=0.001):
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


class MLPscratch():
    """
        MLP model with 1 hidden layer and 256 hidden units.
        input shape for n sample points in a minibatch and d input features,
        X -> (n,d)
        with h hidden units, hidden layer weights & biases
        W(1) -> (d,h)
        b(1) -> (1,h)
        Hidden layer output
        H -> (n,h)
        output layer weights for q classes
        W(2) -> (h,q)
        b(2) -> (1,q)
        output
        O -> (n,q)
    """

    def __init__(self, input_shape, num_classes, num_hiddens= 256, seed_value= None):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.num_hiddens = num_hiddens

        self.mlp_model = None
        self.model = None

        if seed_value is not None:
            set_seed(seed_value)
            self.seed_value = seed_value

    def MLP(self, lr=0.001, h_activation='relu'):
        tf.keras.backend.clear_session()

        self.mlp_model = tf.keras.Sequential([
            tf.keras.Input(shape= self.input_shape),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(self.num_hiddens, activation=h_activation),
            tf.keras.layers.Dense(self.num_classes, activation= 'softmax')
        ])

        self.mlp_model.compile(
            tf.keras.optimizers.Adam(learning_rate= lr),
            loss='sparse_categorical_crossentropy',
            metrics= ['accuracy']
        )

    def deeperMLP(self, lr=0.001, num_hidden1=512, num_hidden2=256, num_hidden3=128, dropout_rate1=0.2, dropout_rate2=0.3, dropout_rate3=0.3, slope=0.1):
        tf.keras.backend.clear_session()

        self.model = tf.keras.Sequential([
            tf.keras.Input(shape= self.input_shape),
            tf.keras.layers.Flatten(),

            tf.keras.layers.Dense(num_hidden1, 
                                  kernel_initializer=tf.keras.initializers.HeNormal(seed=self.seed_value),
                                  kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(negative_slope=slope),
            tf.keras.layers.Dropout(dropout_rate1),

            tf.keras.layers.Dense(num_hidden2, 
                                  kernel_initializer=tf.keras.initializers.HeNormal(seed=self.seed_value),
                                  kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(negative_slope=slope),
            tf.keras.layers.Dropout(dropout_rate2),

            tf.keras.layers.Dense(num_hidden3, 
                                  kernel_initializer=tf.keras.initializers.HeNormal(seed=self.seed_value),
                                  kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(negative_slope=slope),
            tf.keras.layers.Dropout(dropout_rate3),

            tf.keras.layers.Dense(self.num_classes, activation= 'softmax')
        ])

        self.model.compile(
            tf.keras.optimizers.Adam(weight_decay=1e-4, learning_rate= lr),
            loss= 'sparse_categorical_crossentropy',
            metrics= ['accuracy']
        )