from model import Model
from log_reg_model import LogisticRegressionModel
from soft_reg_model import SoftmaxRegressionModel
from data_batcher import DataBatcher
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

epochs = 400
batch_size = 20

batcher = DataBatcher("../normalizedData/trainingSetX.npy", "../normalizedData/trainingSetY.npy", "../normalizedData/testSetX.npy", "../normalizedData/testSetY.npy")
model = Model([40])
log_model = LogisticRegressionModel()
soft_model = SoftmaxRegressionModel()

plot_x = list()
plot_y = list()
plot_y_log = list()
plot_y_soft = list()
with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    epoch_index = 0
    while epoch_index < epochs:
        samples, labels = batcher.get_batch(batch_size)
        model.train_model(session, samples, labels)
        log_model.train_model(session, samples, labels)
        soft_model.train_model(session, samples, labels)
        if batcher.epoch_finished():
            batcher.reset_epoch()
            test_samples, test_labels = batcher.get_test_batch()
            acc = model.get_accuracy(session, test_samples, test_labels)
            acc_log = log_model.get_accuracy(session, test_samples, test_labels)
            acc_soft = soft_model.get_accuracy(session, test_samples, test_labels)
            plot_x.append(epoch_index+1)
            plot_y.append(acc)
            plot_y_log.append(acc_log)
            plot_y_soft.append(acc_soft)
            print("Epoch {} ~ NN: {} | SoftR: {} | LogR: {}".format(epoch_index, acc, acc_soft, acc_log))
            epoch_index += 1

sns.set_style("darkgrid")
plt.xlabel("Epochs")
plt.ylabel("Accuracy on Test Set")
plt.plot(plot_x, plot_y, label="Neural Net")
plt.plot(plot_x, plot_y_log, label="Logistic Regression")
plt.plot(plot_x, plot_y_soft, label="Softmax Regression")
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
plt.show()
