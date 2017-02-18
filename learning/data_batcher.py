import numpy as np
import random

class DataBatcher:
    def __init__(self, train_x_path, train_y_path, test_x_path, test_y_path):
        train_x = np.load(train_x_path)
        train_y = np.load(train_y_path)

        self.samples = list()
        for i in range(len(train_x)):
            self.samples.append((train_x[i], train_y[i]))
        self.epoch_samples = list(self.samples)

        test_x = np.load(test_x_path)
        test_y = np.load(test_y_path)

        self.test_samples = list()
        for i in range(len(test_x)):
            self.test_samples.append((test_x[i], test_y[i]))

    def reset_epoch(self):
        self.epoch_samples = list(self.samples)

    def epoch_finished(self):
        return len(self.epoch_samples) == 0

    def get_batch(self, size):
        if size > len(self.epoch_samples):
            size = len(self.epoch_samples)
        b = [self.epoch_samples.pop(random.randrange(len(self.epoch_samples))) for _ in range(size)]
        sam = list()
        lab = list()
        for s, l in b:
            sam.append(s)
            ohl = np.zeros(2)
            ohl[int(l)] = 1.0
            lab.append(ohl)
        k = np.array(lab).T
        return np.array(sam), np.array(lab)

    def get_test_batch(self):
        sam = list()
        lab = list()
        for s, l in self.test_samples:
            sam.append(s)
            ohl = np.zeros(2)
            ohl[int(l)] = 1.0
            lab.append(ohl)
        return np.array(sam), np.array(lab)
