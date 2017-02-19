import tensorflow as tf
import numpy as np

class Model:
    def __init__(self, hidden_layer_sizes):
        with tf.variable_scope("neural_net_model"):
            self.inputs = tf.placeholder(tf.float32, shape=[None, 30])
            self.labels = tf.placeholder(tf.float32, shape=[None, 2])

            rows = 30
            last_op = self.inputs
            for i in hidden_layer_sizes:
                W = tf.Variable(tf.truncated_normal([rows, i], stddev=0.1))
                b = tf.Variable(tf.constant(0.1, shape=[i]))
                hidden_layer = tf.nn.elu(tf.nn.xw_plus_b(last_op, W, b))
                rows = i
                last_op = hidden_layer

            Wout = tf.Variable(tf.truncated_normal([rows, 2], stddev=0.1))
            bout = tf.Variable(tf.constant(0.1, shape=[2]))
            output_layer = tf.nn.xw_plus_b(last_op, Wout, bout)

            self.probabilities = tf.nn.softmax(output_layer)

            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.probabilities, axis=1), tf.argmax(self.labels, axis=1)), tf.float32))

            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=np.array([0.9, 0.1]) * output_layer, labels=self.labels))
            self.train = tf.train.AdamOptimizer(1e-4).minimize(self.loss)

    def train_model(self, session, inputs, labels):
        session.run(self.train, feed_dict={self.inputs: inputs, self.labels: labels})

    def predict(self, session, inputs):
        return session.run(self.probabilities, feed_dict={self.inputs: inputs})

    def get_loss(self, session, inputs, labels):
        return session.run(self.loss, feed_dict={self.inputs: inputs, self.labels: labels})

    def get_accuracy(self, session, inputs, labels):
        return session.run(self.accuracy, feed_dict={self.inputs: inputs, self.labels: labels})
