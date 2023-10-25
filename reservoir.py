import numpy as np
import tensorflow as tf


class Reservoir_eca(tf.keras.Model):
    def __init__(self, rule=110, time_steps=30, shape=(28, 28)):
        super().__init__()
        self.rule = rule
        self.time_steps = time_steps
        self.shape = shape
        self.size = shape[0] * shape[1]

        self.dmodel = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=self.shape),
            tf.numpy_function(func=self.eca_func, inp=[rule, self.size, time_steps, z], Tout=tf.float32),

        ])
        ### Layers ###
        self.flatten = tf.keras.layers.Flatten(input_shape=self.shape)
        self.pool = tf.keras.layers.AveragePooling1D(pool_size=16)
        self.dense = tf.keras.layers.Dense(10)
        self(np.zeros([28, 28]))

    def ECA_step(self, x, rule_b):
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
    
    def ECA(self, rule, num_cells, time_steps, x_init):
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
            x[i + 1, :] = self.ECA_step(x[i, :], rule_b)
        # Cast the result to float32 before returning
        return x.astype(np.float32)[time_steps-1]
    
    def eca_func(self, rule, num_cells, time_steps, z):
        return self.ECA(int(rule), int(num_cells), int(time_steps), z)

    def __call__(self, x):
        z = self.flatten(x)
        z = tf.reshape(z, (-1, 784))
        # print(z.shape)
        dx = tf.numpy_function(self.eca_func, inp=[self.rule, self.size, self.time_steps, z], Tout=tf.float32)
        dx = tf.reshape(dx, (-1, 784, 1))
        pooled = self.pool(dx)
        return self.dense(pooled)
