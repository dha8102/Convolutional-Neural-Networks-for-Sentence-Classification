import tensorflow as tf
import numpy as np
from config import *


class model_build():
    def __init__(self, sequence_length, num_classes,
                 vocab_size, embeddings):

        # Declaration variables in the class
        self.dropout_prob = tf.placeholder(tf.float32, name="dropout_prob")
        l2_loss = tf.constant(0.0)
        # Input layer
        self.inputX = tf.placeholder(tf.int32, [None, sequence_length], name="inputX")
        self.inputY = tf.placeholder(tf.int32, [None, num_classes], name="inputY")
        print("in class, shape(embeddings) :", np.shape(embeddings))
        self.inputembeddings = tf.constant(embeddings, dtype=tf.float32)
        if model_variation() != 3:

            # Embedding layer
            if model_variation() == 0:
                self.all_embeddings = tf.get_variable(initializer=tf.random_uniform([vocab_size, embedding_dim()], -1.0, 1.0),
                                                      name="inputembeddings")
                #shape=[vocab_size, embedding_dim()],
                #initializer = tf.contrib.layers.variance_scaling_initializer(),
            elif model_variation() == 1:
                self.all_embeddings = tf.get_variable(initializer=self.inputembeddings,
                                                      name="inputembeddings", trainable=False)
            elif model_variation() == 2:
                self.all_embeddings = tf.get_variable(initializer=self.inputembeddings,
                                                      name="inputembeddings")

            self.embedding_lookup = tf.nn.embedding_lookup(self.all_embeddings, self.inputX)
            self.embeddings = tf.expand_dims(self.embedding_lookup, -1, name="embeddings")

            # Filter convolution
            pooled_results = []
            for i, h in enumerate(filter_size()):
                filter_shape = [h, embedding_dim(), 1, num_filters()]
                '''
                filter_weight = tf.get_variable(shape=filter_shape,
                                                initializer=tf.contrib.layers.variance_scaling_initializer(),
                                                name=("filter_weight%d" % i))
                '''
                filter_weight = tf.get_variable(name=("filter_weight%d" % i),
                                                initializer=tf.random.uniform(shape=filter_shape,
                                                                              minval=-0.01,
                                                                              maxval=0.01))
                conv = tf.nn.conv2d(self.embeddings, filter_weight, strides=[1, 1, 1, 1],
                                    padding="VALID", name="conv")

                # add bias and non-linearity
                filter_bias = tf.Variable(tf.constant(0.1, shape=[num_filters()]),
                                          name="filter_bias")
                conv_relu = tf.nn.relu(tf.nn.bias_add(conv, filter_bias), name="conv_relu")
                self.pooling = tf.nn.max_pool(conv_relu,
                                              ksize=[1, sequence_length - h + 1, 1, 1],
                                              strides=[1, 1, 1, 1],
                                              padding="VALID",
                                              name="pooling")
                pooled_results.append(self.pooling)
            total_filter_num = num_filters() * len(filter_size())

        else:  # multichannel
            self.all_embeddings1 = tf.get_variable(initializer=self.inputembeddings,
                                                   name="inputembeddings1", trainable=False)

            self.all_embeddings2 = tf.get_variable(initializer=self.inputembeddings,
                                                   name="inputembeddings2")

            self.embedding_lookup1 = tf.nn.embedding_lookup(self.all_embeddings1, self.inputX)
            self.embedding_lookup2 = tf.nn.embedding_lookup(self.all_embeddings2, self.inputX)

            self.embeddings1 = tf.expand_dims(self.embedding_lookup1, -1, name="embeddings1")
            self.embeddings2 = tf.expand_dims(self.embedding_lookup1, -1, name="embeddings2")

            # Filter Convolution
            pooled_results = []
            for i, h in enumerate(filter_size()):
                filter_shape = [h, embedding_dim(), 1, num_filters()]
                filter_weight = tf.get_variable(shape=filter_shape,
                                                initializer=tf.contrib.layers.variance_scaling_initializer(),
                                                name=("filter_weight%d" % i))
                conv1 = tf.nn.conv2d(self.embeddings1, filter_weight, strides=[1, 1, 1, 1],
                                     padding="VALID", name="conv1")

                # add bias and non-linearity
                filter_bias1 = tf.Variable(tf.constant(0.1, shape=[num_filters()]),
                                           name="filter_bias1")
                conv1_relu = tf.nn.relu(tf.nn.bias_add(conv1, filter_bias1), name="conv1_relu")
                self.pooling1 = tf.nn.max_pool(conv1_relu,
                                               ksize=[1, sequence_length - h + 1, 1, 1],
                                               strides=[1, 1, 1, 1],
                                               padding="VALID",
                                               name="pooling1")
                pooled_results.append(self.pooling1)

            for i, h in enumerate(filter_size()):
                filter_shape = [h, embedding_dim(), 1, num_filters()]
                filter_weight = tf.get_variable(shape=filter_shape,
                                                initializer=tf.contrib.layers.variance_scaling_initializer(),
                                                name=("filter_weight%d" % (i + 3)))
                conv2 = tf.nn.conv2d(self.embeddings2, filter_weight, strides=[1, 1, 1, 1],
                                     padding="VALID", name="conv2")

                # add bias and non-linearity
                filter_bias2 = tf.Variable(tf.constant(0.1, shape=[num_filters()]),
                                           name="filter_bias2")
                conv_relu2 = tf.nn.relu(tf.nn.bias_add(conv2, filter_bias2), name="conv_relu2")
                self.pooling2 = tf.nn.max_pool(conv_relu2,
                                               ksize=[1, sequence_length - h + 1, 1, 1],
                                               strides=[1, 1, 1, 1],
                                               padding="VALID",
                                               name="pooling2")
                pooled_results.append(self.pooling2)
            total_filter_num = num_filters() * len(filter_size()) * 2

        # making penultimate layer
        self.concat = tf.concat(pooled_results, 3)
        self.concat_reshape = tf.reshape(self.concat, [-1, total_filter_num])

        # add dropout
        self.dropout_layer = tf.nn.dropout(self.concat_reshape, self.dropout_prob)

        if model_variation() != 3:
            '''
            final_weight = tf.get_variable("final_weight", shape=[embedding_dim(), num_classes],
                                           initializer=tf.contrib.layers.variance_scaling_initializer())
            '''
            final_weight = tf.get_variable("final_weight",
                                           initializer=tf.random.uniform(shape=[embedding_dim(), num_classes],
                                                                         minval=-0.01,maxval=0.01))
        else:
            final_weight = tf.get_variable("final_weight", shape=[embedding_dim() * 2, num_classes],
                                           initializer=tf.contrib.layers.variance_scaling_initializer())
        # initializer=tf.contrib.layers.xavier_initializer())
        final_bias = tf.Variable(tf.constant(0.1, shape=[num_classes]))
        self.score = tf.add(tf.matmul(self.dropout_layer, final_weight), final_bias, name="score")
        self.prediction = tf.argmax(self.score, 1, name="prediction")
        l2_loss += tf.nn.l2_loss(final_weight)
        l2_loss += tf.nn.l2_loss(final_bias)
        self.loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.score,
                                                       labels=self.inputY)) + l2_loss * l2_lambda()
        self.correct = tf.equal(self.prediction, tf.argmax(self.inputY, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct, "float"), name="accuracy")
