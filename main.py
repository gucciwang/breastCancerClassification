from model import Model
from data_batcher import DataBatcher
import tensorflow as tf

epochs = 50
batch_size = 20

batcher = DataBatcher("data.txt")
model = Model(15)

with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    epoch_index = 0
    while epoch_index < epochs:
        samples, labels = batcher.get_batch(batch_size)
        print(samples)
        model.train_model(session, samples, labels)
        if batcher.epoch_finished():
            batcher.reset_epoch()
            test_samples, test_labels = batcher.get_test_batch()
            loss = model.get_loss(session, test_samples, test_labels)
            print("Epoch {} -> {}".format(epoch_index, loss))
            epoch_index += 1
