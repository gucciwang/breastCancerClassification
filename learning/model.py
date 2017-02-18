import tensorflow as tf

class Model:
    def __init__(self, hidden_layer_size):
        self.inputs = tf.placeholder(tf.float32, shape=[None, 9])
        self.labels = tf.placeholder(tf.float32, shape=[None, 2])

        W1 = tf.Variable(tf.truncated_normal([9, hidden_layer_size], stddev=0.1))
        b1 = tf.Variable(tf.constant(0.1, shape=[hidden_layer_size]))
        hidden_layer = tf.nn.tanh(tf.nn.xw_plus_b(self.inputs, W1, b1))

        Wout = tf.Variable(tf.truncated_normal([hidden_layer_size, 2], stddev=0.1))
        bout = tf.Variable(tf.constant(0.1, shape=[2]))
        output_layer = tf.nn.xw_plus_b(hidden_layer, Wout, bout)

        self.probabilities = tf.nn.softmax(output_layer)

        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output_layer, labels=self.labels))
        self.train = tf.train.AdamOptimizer(1e-4).minimize(self.loss)

    def train_model(self, session, inputs, labels):
        session.run(self.train, feed_dict={self.inputs: inputs, self.labels: labels})

    def predict(self, session, inputs):
        return session.run(self.probabilities, feed_dict={self.inputs: inputs})

    def get_loss(self, session, inputs, labels):
        return session.run(self.loss, feed_dict={self.inputs: inputs, self.labels: labels})
