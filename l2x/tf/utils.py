# -*- coding: utf-8 -*-

import json

import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import Layer, Conv1D, GlobalMaxPooling1D, Embedding, Dense, Dropout
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing import sequence

import logging

logger = logging.getLogger(__name__)


class IMLESubsetkLayer(tf.keras.layers.Layer):
    def __init__(self, k, _tau=30.0, _lambda=1000.0):
        super(IMLESubsetkLayer, self).__init__()

        self.k = k
        self._tau = _tau
        self._lambda = _lambda
        self.samples = None

    def sample_gumbel(self, shape, eps=1e-20):
        U = tf.random.uniform(shape, minval=0, maxval=1)
        return -tf.math.log(-tf.math.log(U + eps) + eps)

    def sample_discrete(self, logits):
        gumbel_softmax_sample = logits + self.sample_gumbel(tf.shape(logits))
        threshold = tf.expand_dims(tf.nn.top_k(gumbel_softmax_sample, self.k, sorted=True)[0][:, -1], -1)
        y = tf.cast(tf.greater_equal(gumbel_softmax_sample, threshold), tf.float32)
        return y

    @tf.function
    def sample_gumbel_k(self, shape):
        s = tf.map_fn(fn=lambda t: tf.random.gamma(shape, 1.0 / self.k, self.k / t),
                      elems=tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]))
        # now add the samples
        s = tf.reduce_sum(s, 0)
        # the log(m) term
        s = s - tf.math.log(10.0)
        # divide by k --> each s[c] has k samples whose sum is distributed as Gumbel(0, 1)
        s = self._tau * (s / self.k)
        return s

    # @tf.function
    def sample_discrete_2(self, logits):
        self.samples = self.sample_gumbel_k(tf.shape(logits))
        gumbel_softmax_sample = logits + self.samples
        threshold = tf.expand_dims(tf.nn.top_k(gumbel_softmax_sample, self.k, sorted=True)[0][:, -1], -1)
        y = tf.cast(tf.greater_equal(gumbel_softmax_sample, threshold), tf.float32)
        return y

    # @tf.function
    def sample_discrete_2_reuse(self, logits):
        gumbel_softmax_sample = logits + self.samples
        threshold = tf.expand_dims(tf.nn.top_k(gumbel_softmax_sample, self.k, sorted=True)[0][:, -1], -1)
        y = tf.cast(tf.greater_equal(gumbel_softmax_sample, threshold), tf.float32)
        return y

    @tf.custom_gradient
    def gumbel_topk_new(self, logits):
        # we compute a map state for the distribution
        # we also store the sample for later
        z_train = self.sample_discrete_2(logits)
        threshold = tf.expand_dims(tf.nn.top_k(logits, self.k, sorted=True)[0][:, -1], -1)
        z_test = tf.cast(tf.greater_equal(logits, threshold), tf.float32)
        z_output = K.in_train_phase(z_train, z_test)

        def custom_grad(dy):
            # we perturb (implicit diff) and then resuse sample for perturb and MAP
            map_dy = self.sample_discrete_2_reuse(logits - (self._lambda * dy))
            # we now compute the gradients as the difference (I-MLE gradients)
            grad = tf.math.subtract(z_train, map_dy)
            # for the straight-through estimator, simply use the following line
            # return dy, k
            return grad

        return z_output, custom_grad

    def call(self, logits):
        logits = tf.squeeze(logits, -1)  # [batchsize, d]
        y = self.gumbel_topk_new(logits)
        y = tf.expand_dims(y, -1)  # [batchsize, d, 1]
        return y

    def get_config(self):
        cfg = super().get_config()
        return cfg


EPSILON = np.finfo(tf.float32.as_numpy_dtype).tiny


def gumbel_keys(w):
    # sample some gumbels
    uniform = tf.random.uniform(tf.shape(w), minval=EPSILON, maxval=1.0)
    z = tf.math.log(-tf.math.log(uniform))
    w = w + z
    return w


def continuous_topk(w, k, t, separate=False):
    khot_list = []
    onehot_approx = tf.zeros_like(w, dtype=tf.float32)
    for i in range(k):
        khot_mask = tf.maximum(1.0 - onehot_approx, EPSILON)
        w += tf.math.log(khot_mask)
        onehot_approx = tf.nn.softmax(w / t, axis=-1)
        khot_list.append(onehot_approx)
    return khot_list if separate else tf.reduce_sum(khot_list, 0)


def sample_subset(w, k, t=0.1):
    '''
    Args:
        w (Tensor): Float Tensor of weights for each element. In gumbel mode
            these are interpreted as log probabilities
        k (int): number of elements in the subset sample
        t (float): temperature of the softmax
    '''
    w = gumbel_keys(w)
    return continuous_topk(w, k, t)


class SampleSubset(Layer):
    """
    Layer for continuous approx of subset sampling
    """

    def __init__(self, tau0, k, **kwargs):
        self.tau0 = tau0
        self.k = k
        super(SampleSubset, self).__init__(**kwargs)

    def call(self, logits):
        # logits: [BATCH_SIZE, d, 1]
        logits = tf.squeeze(logits, 2)
        samples = sample_subset(logits, self.k, self.tau0)

        # Explanation Stage output.
        threshold = tf.expand_dims(tf.nn.top_k(logits, self.k, sorted=True)[0][:, -1], -1)
        discrete_logits = tf.cast(tf.greater_equal(logits, threshold), tf.float32)
        output = K.in_train_phase(samples, discrete_logits)
        return tf.expand_dims(output, -1)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        cfg = super().get_config()
        return cfg


class Concatenate(Layer):
    """
    Layer for concatenation.

    """

    def __init__(self, **kwargs):
        super(Concatenate, self).__init__(**kwargs)

    def call(self, inputs):
        input1, input2 = inputs
        input1 = tf.expand_dims(input1, axis=-2)  # [batchsize, 1, input1_dim]
        dim1 = int(input2.get_shape()[1])
        input1 = tf.tile(input1, [1, dim1, 1])
        return tf.concat([input1, input2], axis=-1)

    def compute_output_shape(self, input_shapes):
        input_shape1, input_shape2 = input_shapes
        input_shape = list(input_shape2)
        input_shape[-1] = int(input_shape[-1]) + int(input_shape1[-1])
        input_shape[-2] = int(input_shape[-2])
        return tuple(input_shape)


class Sample_Concrete(Layer):
    """
    Layer for sample Concrete / Gumbel-Softmax variables.

    """

    def __init__(self, tau0, k, **kwargs):
        self.tau0 = tau0
        self.k = k
        super(Sample_Concrete, self).__init__(**kwargs)

    def call(self, logits):
        # logits: [batch_size, d, 1]
        logits_ = K.permute_dimensions(logits, (0, 2, 1))  # [batch_size, 1, d]

        d = int(logits_.get_shape()[2])

        uniform = tf.random.uniform(shape=tf.shape(logits_), minval=0.0, maxval=1.0)
        gumbel = - K.log(-K.log(uniform))
        noisy_logits = (gumbel + logits_) / self.tau0
        samples = K.softmax(noisy_logits)
        samples = K.max(samples, axis=1)
        logits = tf.reshape(logits, [-1, d])
        threshold = tf.expand_dims(tf.nn.top_k(logits, self.k, sorted=True)[0][:, -1], -1)
        discrete_logits = tf.cast(tf.greater_equal(logits, threshold), tf.float32)

        output = K.in_train_phase(samples, discrete_logits)
        return tf.expand_dims(output, -1)

    def get_config(self):
        cfg = super().get_config()
        return cfg

    def compute_output_shape(self, input_shape):
        return input_shape


def construct_gumbel_selector(X_ph, num_words, embedding_dims, embedding_matrix, kernel_size, maxlen):
    """
    Build the L2X model for selecting words.
    """
    emb_layer = Embedding(num_words,
                          embedding_dims,
                          weights=[embedding_matrix],
                          input_length=maxlen,
                          trainable=False,
                          name='emb_gumbel')

    # XXX (None, 350, 200)
    emb = emb_layer(X_ph)  # (350, 200)

    # net = Dropout(0.2, name = 'dropout_gumbel')(emb)# this is not used in the L2X experiments
    net = emb

    # 100 here should be "filters"
    conv_layer = Conv1D(100, kernel_size, padding='same', activation='relu', strides=1, name='conv1_gumbel')
    # XXX (None, 350, 100)
    first_layer = conv_layer(net)

    # global info
    pooling_layer = GlobalMaxPooling1D(name='new_global_max_pooling1d_1')
    # XXX (None, 100)
    net_new = pooling_layer(first_layer)

    dense_layer = Dense(100, name='new_dense_1', activation='relu')
    # XXX (None, 100)
    global_info = dense_layer(net_new)

    # local info
    conv_layer = Conv1D(100, kernel_size, padding='same', activation='relu', strides=1, name='conv2_gumbel')
    # XXX (None, 350, 100)
    net = conv_layer(first_layer)

    conv_layer = Conv1D(100, kernel_size, padding='same', activation='relu', strides=1, name='conv3_gumbel')
    # XXX (None, 350, 100)
    local_info = conv_layer(net)

    concat_layer = Concatenate()
    # XXX (None, 350, 200)
    combined = concat_layer([global_info, local_info])

    dropout_layer = Dropout(0.2, name='new_dropout_2')
    # XXX (None, 350, 200)
    net = dropout_layer(combined)

    conv_layer = Conv1D(100, 1, padding='same', activation='relu', strides=1, name='conv_last_gumbel')
    # XXX (None, 350, 100)
    net = conv_layer(net)

    conv_layer = Conv1D(1, 1, padding='same', activation=None, strides=1, name='conv4_gumbel')
    # XXX (None, 350, 1)
    logits_T = conv_layer(net)

    return logits_T


def subset_precision(modelTestInput, aspect, id_to_word, word_to_id, select_k):
    data = []
    num_annotated_reviews = 0
    with open("data/annotations.json") as fin:
        for line in fin:
            item = json.loads(line)
            data.append(item)
            num_annotated_reviews = num_annotated_reviews + 1

    selected_word_counter = 0
    correct_selected_counter = 0

    for anotr in range(num_annotated_reviews):
        ranges = data[anotr][str(aspect)]  # the aspect id
        text_list = data[anotr]['x']
        review_length = len(text_list)

        list_test = []
        tokenid_list = [word_to_id.get(token, 0) for token in text_list]
        list_test.append(tokenid_list)

        X_test_subset = np.asarray(list_test)
        X_test_subset = sequence.pad_sequences(X_test_subset, maxlen=350)

        # X_test_subset = pad_sequences(list_test, max_len=350)

        prediction = modelTestInput.predict(X_test_subset)

        # print(prediction.shape)

        prediction = tf.squeeze(prediction, -1)

        # import sys
        # sys.exit(0)

        x_val_selected = prediction[0] * X_test_subset

        selected_words = np.vectorize(id_to_word.get)(x_val_selected)[0][-review_length:]
        selected_nonpadding_word_counter = 0

        for i, w in enumerate(selected_words):
            if w != '<PAD>':  # we are nice to the L2X approach by only considering selected non-pad tokens
                selected_nonpadding_word_counter = selected_nonpadding_word_counter + 1
                for r in ranges:
                    rl = list(r)
                    if i in range(rl[0], rl[1]):
                        correct_selected_counter = correct_selected_counter + 1
        # we make sure that we select at least 10 non-padding words
        # if we have more than select_k non-padding words selected, we allow it but count that in
        selected_word_counter = selected_word_counter + max(selected_nonpadding_word_counter, select_k)

    return correct_selected_counter / selected_word_counter
