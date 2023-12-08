import tensorflow as tf
import numpy as np
import math
import pickle

class AllConvModel(tf.Module):
    
    def __init__(self, model_path):
        super().__init__()
        with open(model_path+".info","rb") as f:
            model_info = pickle.load(f)
        
        num_classes = model_info["num_classes"]
        input_shape = model_info["input_shape"]
        num_filters = model_info["num_filters"]
    
        conv_args = dict(
            activation=tf.nn.leaky_relu,
            kernel_size=3,
            padding='same')
        
        self.layers = []
        log_resolution = int(round(
            math.log(input_shape[0]) / math.log(2)))
        for scale in range(log_resolution - 2):
            self.layers.append(tf.keras.layers.Conv2D(num_filters << scale, **conv_args))
            self.layers.append(tf.keras.layers.Conv2D(num_filters << (scale + 1), **conv_args))
            self.layers.append(tf.keras.layers.AveragePooling2D((2, 2)))
        self.layers.append(tf.keras.layers.Conv2D(num_classes, kernel_size=3, padding='same'))
        self.layers.append(tf.keras.layers.GlobalAveragePooling2D())
        self.layers.append(tf.keras.layers.Softmax())

        # call inference to instantiate variables
        self(tf.zeros((1,) + tuple(input_shape)), training=False)

    def __call__(self, x, training=False):
        del training  
        for layer in self.layers:
            x = layer(x)
        return x


def load_cifar10_model(model_path):
    model = AllConvModel(model_path)
    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint.restore(model_path)
    return model