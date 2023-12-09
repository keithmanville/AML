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


def augment(x, y):
    x_shape = tf.shape(x)
    x = tf.image.random_flip_left_right(x)
    x = tf.pad(x, [[0] * 2, [4] * 2, [4] * 2, [0] * 2], mode='REFLECT')
    return tf.image.random_crop(x, x_shape), y


def augment_strong(image,
                   strength=.5):
    # Color jitter taken from SimCLR implementation

    brightness = 0.8 * strength
    contrast = 0.8 * strength
    saturation = 0.8 * strength
    hue = 0.2 * strength
    def apply_transform(i, x):
        def brightness_foo():
            return tf.image.random_brightness(x, max_delta=brightness)
        
        def contrast_foo():
            return tf.image.random_contrast(x, lower=1-contrast, upper=1+contrast)
        
        def saturation_foo():
            return tf.image.random_saturation(
                x, lower=1-saturation, upper=1+saturation)
  
        def hue_foo():
            return tf.image.random_hue(x, max_delta=hue)
        
        x = tf.cond(tf.less(i, 2),
                    lambda: tf.cond(tf.less(i, 1), brightness_foo, contrast_foo),
                    lambda: tf.cond(tf.less(i, 3), saturation_foo, hue_foo))
        return x
  
    perm = tf.random.shuffle(tf.range(4))
    for i in range(4):
        image = apply_transform(perm[i], image)
        image = tf.clip_by_value(image, 0., 1.)
    return augment(image, None)[0]


def augment_strong_np(image, strength=.5):
    return augment_strong(image, strength).numpy()
