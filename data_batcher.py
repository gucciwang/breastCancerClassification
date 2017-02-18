import numpy as np
import random

class DataBatcher:
    def __init__(self, data_file_path):
        data = np.genfromtxt(data_file_path, delimiter=",")
        data = np.delete(data, 0, 1)

        labels = data[:,-1]
        labels = (labels == 4).astype(float)
        data = np.delete(data, -1, 1)

        self.samples = list()
        for i in range(len(data) - int(0.1 * len(data))):
            self.samples.append((data[i], labels[i]))
        self.epoch_samples = list(self.samples)

        self.test_samples = list()
        for i in range(len(data) - int(0.1 * len(data)), len(data)):
            self.test_samples.append((data[i], labels[i]))

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
