import os
import numpy as np
import tensorflow as tf
import random
import pickle
import matplotlib.pyplot as plt
from cnn_model import model_build
from config import *

os.environ["CUDA_VISIBLE_DEVICES"] = '1'  # 할당할 GPU번호 입력


def dev_split(x_train, y_train, thiscv, devlen):
    x_dev = []
    y_dev = []
    new_x_train = []
    new_y_train = []
    for i in range(len(x_train)):
        if thiscv * devlen <= i < (thiscv + 1) * devlen:
            x_dev.append(x_train[i])
            y_dev.append(y_train[i])
        else:
            new_x_train.append(x_train[i])
            new_y_train.append(y_train[i])
    # print(np.shape(new_x_train),np.shape(new_y_train),np.shape(x_dev),np.shape(y_dev))
    return new_x_train, new_y_train, x_dev, y_dev


def making_batch(x_train, y_train, batch_idx, change):
    x_batch = []
    y_batch = []
    batch_cnt = 0
    while batch_cnt < batch_size():
        x_batch.append(x_train[batch_idx])
        y_batch.append(y_train[batch_idx])
        batch_idx += 1
        if batch_idx == len(x_train):
            batch_idx = 0
            change = True
        batch_cnt += 1
    # print(np.shape(x_train), np.shape(y_train))
    return x_batch, y_batch, batch_idx, change


# shuffle data
def shuffle_data(x, y):
    shuffle_x = []
    shuffle_y = []
    shuffle_list = list(range(len(x)))
    random.shuffle(shuffle_list)
    for i in shuffle_list:
        shuffle_x.append(x[i])
        shuffle_y.append(y[i])
    return shuffle_x, shuffle_y


if __name__ == "__main__":

    print("train data loading...")
    # train data load
    f = open("TREC_original_data.bin", "rb")
    total_data = pickle.load(f)
    f.close()

    x_train = total_data[0]
    x_test = total_data[1]
    y_train = total_data[2]
    y_test = total_data[3]
    voca = total_data[4]
    embeddings = np.array(total_data[5])
    print("embedding_shape =", np.shape(embeddings))
    print("voca_len = ", len(voca))

    # model build
    cnn = model_build(sequence_length=len(x_train[0]), num_classes=len(y_train[0]),
                      vocab_size=len(voca), embeddings=embeddings)

    sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
    saver = tf.train.Saver(tf.global_variables())
    global_step = tf.Variable(0, trainable=False, name='global_step')
    optimizer = tf.train.AdamOptimizer(1e-3)
    grads_and_vars = optimizer.compute_gradients(cnn.loss)
    for i, (g, v) in enumerate(grads_and_vars):
        if g is not None:
            grads_and_vars[i] = (tf.clip_by_norm(g, l2_lambda()), v)

    train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

    ckpt = tf.train.get_checkpoint_state('./TRECmodel')
    devlen = int(len(x_train) * dev_ratio())
    test_accuracy_ave = 0
    test_accuracy_list = []
    each_accuracy = []
    each_loss = []
    if model_variation() == 0:
        print("Model : rand")
    elif model_variation() == 1:
        print("Model : static")
    elif model_variation() == 2:
        print("Model : non-static")
    elif model_variation() == 3:
        print("Model : multi-channel")
    for i in range(3):
        this_x_train, this_y_train, x_dev, y_dev = dev_split(x_train, y_train, i, devlen)
        sess.run(tf.global_variables_initializer())
        if i == 0:
            print("train data size :", len(this_x_train))
            print("development data size :", len(x_dev))
            print("test data size :", len(x_test))
        batch_idx = 0
        accuracy_cnt = 0
        epochcnt = 0
        stopcnt = 0

        print("Epoch :", epochcnt)
        while True:

            epochchange = False
            x_batch, y_batch, batch_idx, change = making_batch(this_x_train, this_y_train, batch_idx, epochchange)
            if change:
                this_x_train, this_y_train = shuffle_data(this_x_train, this_y_train)
                epochcnt += 1
                print("Epoch :", epochcnt)

            feed_dict = {cnn.inputX: x_batch,
                         cnn.inputY: y_batch,
                         cnn.dropout_prob: dropout_prob()}
            _, step, train_loss, = sess.run([train_op, global_step, cnn.loss],
                                            feed_dict=feed_dict)

            if step % 10 == 0:
                feed_dict = {cnn.inputX: x_dev,
                             cnn.inputY: y_dev,
                             cnn.dropout_prob: 1.0}
                step, dev_loss, accuracy = sess.run([global_step, cnn.loss, cnn.accuracy],
                                                    feed_dict=feed_dict)
                each_accuracy.append(accuracy)
                each_loss.append(dev_loss)
                print("step:", step, "train_loss :", train_loss, "dev_loss:", dev_loss, "accuracy:", accuracy)
                if accuracy > 0.91: stopcnt += 1
                if (epochcnt > 10 and stopcnt > 2) or epochcnt > 7:
                    print(i + 1, "th data trained!!")
                    saver.save(sess, './TRECmodel/TREC-%d.ckpt' % (i + 1), global_step=global_step)
                    feed_dict = {cnn.inputX: x_test,
                                 cnn.inputY: y_test,
                                 cnn.dropout_prob: 1.0}
                    test_accuracy = sess.run(cnn.accuracy, feed_dict=feed_dict)

                    print("test accuracy :", test_accuracy)
                    test_accuracy_ave += test_accuracy
                    test_accuracy_list.append(test_accuracy)
                    break

    test_accuracy_ave /= 3
    print("accuracy for each cv : ", end="")
    for i in test_accuracy_list:
        print("%.3f" % i, end=", ")
    print("final accuracy :", test_accuracy_ave)
    '''
    plt.plot(each_accuracy)
    plt.savefig("each_accuracy.png")
    plt.show()

    plt.plot(each_loss)
    plt.savefig("each_loss.png")
    plt.show()
    '''