# obtained from https://www.appsloveworld.com/tensorflow/9/tensorflow-periodic-padding

import numpy as np
import tensorflow as tf

class PeriodicPadding2D(tf.keras.layers.Layer):
    def __init__(self, padding=1, **kwargs):
        super(PeriodicPadding2D, self).__init__(**kwargs)
        self.padding = padding

    def build(self, input_shape):
        super(PeriodicPadding2D, self).build(input_shape)

    def call(self, x):
        # Extract the padding dimensions
        p = self.padding
        # assemble padded x from slices
        #            tl,tc,tr
        # padded_x = ml,mc,mr
        #            bl,bc,br
        top_left = x[:, -p:, -p:] # top left
        top_center = x[:, -p:, :] # top center
        top_right = x[:, -p:, :p] # top right
        middle_left = x[:, :, -p:] # middle left
        middle_center = x # middle center
        middle_right = x[:, :, :p] # middle right
        bottom_left = x[:, :p, -p:] # bottom left
        bottom_center = x[:, :p, :] # bottom center
        bottom_right = x[:, :p, :p] # bottom right
        top = tf.concat([top_left, top_center, top_right], axis=2)
        middle = tf.concat([middle_left, middle_center, middle_right], axis=2)
        bottom = tf.concat([bottom_left, bottom_center, bottom_right], axis=2)
        padded_x = tf.concat([top, middle, bottom], axis=1)

        return padded_x

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1] + 2 * self.padding[0], input_shape[2] + 2 * self.padding[1], input_shape[3])

    def get_config(self):
        config = super(PeriodicPadding2D, self).get_config()
        config.update({'padding': self.padding})
        return config
    
class GOL_layer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(GOL_layer, self).__init__(**kwargs)
        # self.timesteps = timesteps
        self.kernel = self.build_kernel()
    
    def build_kernel(self):
        kernel = tf.constant([[1, 1, 1],
                     [1, 9, 1],
                     [1, 1, 1]], dtype=tf.float32)
        kernel = kernel[..., None, None]
        return kernel

    def build(self, input_shape):
        super(GOL_layer, self).build(input_shape)

    def GOL(self, x):
        mask1 = (x == 3) | (x == 11) | (x ==12)
        return tf.cast(mask1, tf.int32)
    
    def convolve(self, x):
        return tf.nn.conv2d(x, self.kernel, padding="VALID", strides=[1, 1, 1, 1])
    
    def call(self, x):
        x = tf.cast(x, tf.float32)
        # for i in tf.range(self.timesteps):
        x = self.convolve(x)
        x = self.GOL(x)
        return x
    
class gol_reservoir_model(tf.keras.Model):
    def __init__(self):
        super(gol_reservoir_model, self).__init__()
        # self.inputs = tf.keras.Input(shape=(28, 28, 1), dtype=tf.int32)
        self.padding = PeriodicPadding2D()
        self.gol = GOL_layer()
        self.flatten = tf.keras.layers.Flatten()
        self.avg_pool = tf.keras.layers.AveragePooling2D(pool_size=(5,5))
        
        self.dense = tf.keras.layers.Dense(10)
        self(tf.zeros([1, 28, 28, 1], dtype=tf.int32))

    def __call__(self, x, **kwargs):
        x = tf.cast(x, tf.int32)
        x = self.padding(x)
        x = self.gol(x)
        # x = self.avg_pool(x)
        x = self.flatten(x)
        x = tf.reshape(x, (x.shape[0], 784))
        return self.dense(x)

    
class gol_reservoir_model_better(tf.keras.Model):
    def __init__(self, name="rECA", timesteps=2):
        super(gol_reservoir_model_better, self).__init__(name=name)
        self.timesteps = timesteps
        self.input_layer = tf.keras.layers.InputLayer(input_shape=(28, 28, 1), dtype=tf.int32)
        self.padding = PeriodicPadding2D()
        self.gol_layer = GOL_layer()
        self.avg_pool = tf.keras.layers.AveragePooling2D(pool_size=(5,5))
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(20, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(10)

    def call(self, inputs):
        x = self.input_layer(inputs)
        # for i in range(self.timesteps):
        x = self.padding(x)
        x = self.gol_layer(x)
        x = self.avg_pool(tf.cast(x, tf.float32))
        x = self.flatten(x)
        x = tf.reshape(x, (inputs.shape[0], x.shape[-1]))
        x = self.dense1(x)
        x = self.dense2(x)
        return x

class no_reservoir(tf.keras.Model):
    def __init__(self, name="rECA", timesteps=2):
        super(no_reservoir, self).__init__(name=name)
        self.input_layer = tf.keras.layers.InputLayer(input_shape=(28, 28, 1), dtype=tf.int32)
        # self.padding = PeriodicPadding2D()
        # self.gol_layer = GOL_layer()
        self.avg_pool = tf.keras.layers.AveragePooling2D(pool_size=(5,5))
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(20, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(10)

    def call(self, inputs):
        x = self.input_layer(inputs)
        # for i in range(self.timesteps):
        #     x = self.padding(x)
        #     x = self.gol_layer(x)
        x = self.avg_pool(tf.cast(x, tf.float32))
        x = self.flatten(x)
        x = tf.reshape(x, (inputs.shape[0], x.shape[-1]))
        x = self.dense1(x)
        x = self.dense2(x)
        return x

    

