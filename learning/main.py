from model import Model
from data_batcher import DataBatcher
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

epochs = 500
batch_size = 20

batcher = DataBatcher("../normalizedData/trainingSetX.npy", "../normalizedData/trainingSetY.npy", "../normalizedData/testSetX.npy", "../normalizedData/testSetY.npy")
model = Model([40])

plot_x = list()
plot_y = list()
with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    epoch_index = 0
    while epoch_index < epochs:
        samples, labels = batcher.get_batch(batch_size)
        model.train_model(session, samples, labels)
        if batcher.epoch_finished():
            batcher.reset_epoch()
            test_samples, test_labels = batcher.get_test_batch()
            # loss = model.get_loss(session, test_samples, test_labels)
            acc = model.get_accuracy(session, test_samples, test_labels)
            plot_x.append(epoch_index+1)
            plot_y.append(acc)
            print("Epoch {} -> {}".format(epoch_index, acc))
            epoch_index += 1

sns.set_style("darkgrid")
plt.plot(plot_x, plot_y)
plt.show()
