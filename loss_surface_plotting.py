import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def setupPlot():
    x = np.linspace(-1.0, 1.0, 50)
    y = np.linspace(-1.0, 1.0, 50)

    X,Y = np.meshgrid(x,y)

    Z = np.zeros_like(X)

    return X,Y,Z

class PlotLoss():
    def __init__(self, model, data, batch_size=256):
        self.model = model
        self.w, self.b = model.tf_model.layers[-1].get_weights()
        self.loss_fn = tf.keras.losses.sparse_categorical_crossentropy

        self.batch_x, self.batch_y = next(iter(data.unbatch().batch(batch_size)))
    

    def calculateLoss(self, cls_idx=0):
        #choosing 2 indices to plot the loss surface
        flat_w = self.w[:,cls_idx]
        top_indices = np.argsort(np.abs(flat_w))
        i1,i2 = top_indices[:2]

        X,Y,Z = setupPlot()

        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                w_temp = self.w.copy()
                b_org = self.b
                w_temp[i1,cls_idx] = X[i,j]
                w_temp[i2,cls_idx] = Y[i,j]

                #recomputing loss
                logits = tf.matmul(tf.reshape(self.batch_x, [self.batch_x.shape[0],-1]), w_temp) + b_org  #-1 tells tf to do the math to keep the num of elements same
                loss = self.loss_fn(self.batch_y, tf.nn.softmax(logits))
                Z[i,j] = tf.reduce_mean(loss).numpy()

        return X,Y,Z