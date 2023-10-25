import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from reservoir import Reservoir_eca

class BinaryRandomProjection(tf.keras.layers.Layer):
    def __init__(self, output_dim, seed=None, **kwargs):
        self.output_dim = output_dim
        self.seed = seed
        super(BinaryRandomProjection, self).__init__(**kwargs)

    def build(self, input_shape):
        # Initialize the weights randomly
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(input_shape[1], self.output_dim),
                                      initializer=tf.keras.initializers.RandomNormal(seed=self.seed), 
                                      trainable=False)  # We set trainable=False to keep the weights constant

 

    def call(self, inputs):
        projected = tf.matmul(inputs, self.kernel)
        return tf.cast(tf.greater(projected, 0.5), tf.float32)
    

def main():

    # parameters
    num_cells = 784
    time_steps = 100
    rule = 30

    # =============================================================================
    # LOAD AND PREPROCESS DATA
    # -=============================================================================

    # Load MNIST dataset and preprocess
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # convert to binary images
    bin_x_train = np.where(x_train != 0, 1, x_train)
    bin_x_test = np.where(x_test != 0, 1, x_test)

    # load model
    model = create_model()
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.001),
              loss=loss_fn,
              metrics=['accuracy'])
    print(model.summary())

    history = model.fit(bin_x_train[0:999], y_train[0:999], batch_size=1, validation_data=(bin_x_test[0:999], y_test[0:999]), epochs=4)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

def ECA_step(x, rule_b):
    """Compute a single step of an elementary cellular
    automaton."""
    # The columns contains the L, C, R values
    # of all cells.
    y = np.vstack((np.roll(x, 1), x, np.roll(x, -1))).astype(np.int8)
    # We get the LCR pattern numbers between 0 and 7. ???
    u = np.array([[4], [2], [1]])
    z = np.sum(y * u, axis=0).astype(np.int8)
    # We get the patterns given by the rule.
    return rule_b[7 - z]

 

def ECA(rule, num_cells, time_steps, x_init):
    """Simulate an elementary cellular automaton given
    its rule (number between 0 and 255)."""
    # Compute the binary representation of the rule.
    rule_b = np.array(
        [int(_) for _ in np.binary_repr(rule, 8)],
        dtype=np.int8)
    x = np.zeros((time_steps, num_cells), dtype=np.int8)
    x[0, :] = x_init
    # Apply the step function iteratively.
    for i in range(time_steps - 1):
        x[i + 1, :] = ECA_step(x[i, :], rule_b)
    # Cast the result to float32 before returning
    return x.astype(np.float32)[time_steps-1]


def create_model(rule=30, num_cells=784, time_steps=100):
    def eca_func(rule, num_cells, time_steps, z):
        return ECA(int(rule), int(num_cells), int(time_steps), z)
    inputs = tf.keras.Input(shape=(28, 28, 1))
    flatten = tf.keras.layers.Flatten(input_shape=(28, 28))  # input layer
    z = flatten(inputs)
    #bin_proj_layer = BinaryRandomProjection(output_dim=784, seed=0)
    #z = bin_proj_layer(z)
    temp_outputs = tf.numpy_function(func=eca_func, inp=[rule, num_cells, time_steps, z], Tout=tf.float32)
    temp_outputs = tf.cast(temp_outputs, dtype=tf.float32)  # Convert to float32
    temp_outputs = tf.reshape(temp_outputs, (1, 784))
    outputs = tf.keras.layers.Dense(10)(temp_outputs)
    model = keras.Model(inputs=inputs, outputs=outputs, name="rECA")
    return model


if __name__ == "__main__":
    main()