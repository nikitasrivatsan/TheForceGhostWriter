#!/usr/bin/env python

from tensorflow.models.rnn import rnn_cell
import tensorflow as tf
import sys
import numpy as np

BATCH_SIZE = 20
HIDDEN_SIZE = 200
EPOCHS = 10
NUM_STEPS = 200

# data manipulation
data = open("star_wars_vii.txt", 'r').read()
data_size = len(data)
chars = set(data)
num_chars = len(chars)
char_idx = {ch:i for i,ch in enumerate(list(chars))}

def main():

    sess = tf.InteractiveSession()

    print "Building network"

    cell = rnn_cell.BasicLSTMCell(HIDDEN_SIZE)
    initial_state = state = tf.zeros([BATCH_SIZE, cell.state_size], dtype = np.float32)

    words = tf.placeholder(tf.float32, [BATCH_SIZE, NUM_STEPS])
    target_words = tf.placeholder(tf.int32, [BATCH_SIZE, NUM_STEPS])

    W_soft = tf.Variable(tf.truncated_normal([HIDDEN_SIZE, num_chars], stddev = 0.01))
    b_soft = tf.Variable(tf.constant(0.01, shape = [num_chars]))

    loss = tf.Variable(tf.constant(0.01))
    for i in range(0, NUM_STEPS):
        with tf.variable_scope("LSTM" + str(i)):
            output, state = cell(tf.reshape(words[:, i], [BATCH_SIZE, 1]), state)
            prediction = tf.nn.softmax(tf.matmul(output, W_soft) + b_soft)

            for j in range(0, BATCH_SIZE):
                tf.add(loss, tf.slice(prediction, tf.pack([j, target_words[j, i]]), [1, 1]))

    loss /= BATCH_SIZE * NUM_STEPS

    final_state = state

    print "Segmenting input data"

    # segment data into batches
    num_batches = data_size / BATCH_SIZE

    data_idx = np.array([char_idx[ch] for ch in data], dtype = np.int32)
    data_batched = np.zeros([BATCH_SIZE, num_batches], dtype = np.int32)
    for i in range(0, BATCH_SIZE):
        data_batched[i] = data_idx[num_batches * i : num_batches * (i + 1)]

    epoch_size = (num_batches - 1) / NUM_STEPS

    # initialize everything
    sess.run(tf.initialize_all_variables())

    print "Iterating over batches"
    
    # iterate over batches
    current_state = initial_state.eval()
    total_loss = 0.0
    for i in range(0, epoch_size):
        x = data_batched[:, i * NUM_STEPS : (i + 1) * NUM_STEPS]
        y = data_batched[:, i * NUM_STEPS + 1 : (i + 1) * NUM_STEPS + 1]

        current_state, current_loss = sess.run([final_state, loss],
                                               feed_dict = {initial_state : current_state, words : x, target_words : y})

        total_loss += current_loss

    print total_loss

if __name__ == "__main__":
    main()
