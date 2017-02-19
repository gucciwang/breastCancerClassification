import numpy as np

class NaiveBayesModel:
    def __init__(self):


    def train_model(self, session, points):
        session.run(self.train, feed_dict={self.training_inputs: points})

    def get_accuracy(self, session, inputs, labels):
