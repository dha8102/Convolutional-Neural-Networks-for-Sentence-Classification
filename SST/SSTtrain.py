import os
import numpy as np
import tensorflow as tf
import random
import pickle
from cnn_model import model_build
from config import *

os.environ["CUDA_VISIBLE_DEVICES"] = '1'  # 할당할 GPU번호 입력


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

    # f = open("./SST1_original_data.bin", "rb")
    f = open("SST2_original_data.bin", "rb")
    total_data = pickle.load(f)
    f.close()
    # total_data shape : [x_train, x_test, x_dev, y_train, y_test, y_dev, voca, pre_embeddings]

    x_train = total_data[0]
    x_test = total_data[1]
    x_dev = total_data[2]
    y_train = total_data[3]
    y_test = total_data[4]
    y_dev = total_data[5]
    voca = total_data[6]
    embeddings = np.array(total_data[7])

    print("length after preprocessed:", len(x_train) + len(x_test) + len(x_dev))

    # model build
    cnn = model_build(sequence_length=len(x_train[0]), num_classes=len(y_train[0]),
                      vocab_size=len(voca), embeddings=embeddings)

    sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
    saver = tf.train.Saver(tf.global_variables())
    global_step = tf.Variable(0, trainable=False, name='global_step')
    # optimizer = tf.train.AdadeltaOptimizer(learning_rate=0.5,epsilon=1e-06)
    optimizer = tf.train.AdamOptimizer(1e-03)
    grads_and_vars = optimizer.compute_gradients(cnn.loss)
    for i, (g, v) in enumerate(grads_and_vars):
        if g is not None:
            grads_and_vars[i] = (tf.clip_by_norm(g, l2_lambda()), v)

    train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

    ckpt = tf.train.get_checkpoint_state('./SST2model')
    test_accuracy_ave = 0
    test_accuracy_list = []
    each_accuracy = []
    each_loss = []

    print("train data size :", len(x_train))
    print("development data size :", len(x_dev))
    print("test data size :", len(x_test))

    batch_idx = 0
    accuracy_cnt = 0
    epochcnt = 0
    stopcnt = 0
    if model_variation() == 0:
        print("Model : rand")
    elif model_variation() == 1:
        print("Model : static")
    elif model_variation() == 2:
        print("Model : non-static")
    elif model_variation() == 3:
        print("Model : multi-channel")
    print("Epoch :", epochcnt)
    for i in range(cv_num()):
        sess.run(tf.global_variables_initializer())
        epochcnt = 0
        while True:
            epochchange = False
            x_batch, y_batch, batch_idx, change = making_batch(x_train, y_train, batch_idx, epochchange)
            if change:
                x_train, y_train = shuffle_data(x_train, y_train)
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
                if accuracy > 0.84: stopcnt += 1
                if stopcnt > 1:
                    print("train finished!!")
                    feed_dict = {cnn.inputX: x_test,
                                 cnn.inputY: y_test,
                                 cnn.dropout_prob: 1.0}
                    test_accuracy = sess.run(cnn.accuracy, feed_dict=feed_dict)

                    print("test accuracy :", test_accuracy)
                    test_accuracy_ave += test_accuracy
                    test_accuracy_list.append(test_accuracy)
                    stopcnt = 0
                    break
        print("accuracy until now :", test_accuracy_list)
        print("cumulative average of accuracy :", test_accuracy_ave / (i + 1))
        print()
