import tensorflow as tf

class LogisticRegressionModel:
    def __init__(self):
        self.inputs = tf.placeholder(tf.float32, shape=[None, 30])
        self.labels = tf.placeholder(tf.float32, shape=[None, 2])

        Wout = tf.Variable(tf.truncated_normal([30, 2], stddev=0.1))
        bout = tf.Variable(tf.constant(0.1, shape=[2]))
        output_layer = tf.nn.sigmoid(tf.nn.xw_plus_b(self.inputs, Wout, bout))

        self.probabilities = tf.nn.softmax(output_layer)

        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.probabilities, axis=1), tf.argmax(self.labels, axis=1)), tf.float32))

        self.loss = tf.reduce_mean(tf.square(output_layer - self.labels))
        self.train = tf.train.AdamOptimizer(1e-4).minimize(self.loss)

    def train_model(self, session, inputs, labels):
        session.run(self.train, feed_dict={self.inputs: inputs, self.labels: labels})

    def predict(self, session, inputs):
        return session.run(self.probabilities, feed_dict={self.inputs: inputs})

    def get_loss(self, session, inputs, labels):
        return session.run(self.loss, feed_dict={self.inputs: inputs, self.labels: labels})

    def get_accuracy(self, session, inputs, labels):
        return session.run(self.accuracy, feed_dict={self.inputs: inputs, self.labels: labels})
