#!/usr/bin/env python

from tensorflow.models.rnn import rnn_cell
import tensorflow as tf
import sys
import math
import numpy as np
import random

# hyperparameters
BATCH_SIZE = 20
HIDDEN_SIZE = 200
EPOCHS = 5
NUM_STEPS = 100
NUM_SEQUENCES = 2

if len(sys.argv) == 3:
    _, filename, behavior = sys.argv
else:
    print "./lstm.py <filename> [train|test]"
    sys.exit()

# data manipulation
data = open(filename, 'r').read()
data_size = len(data)
chars = list(set(data))
num_chars = len(chars)
char_idx = {ch:i for i,ch in enumerate(chars)}

def main():

    sess = tf.InteractiveSession()

    print "Building network"

    cell = rnn_cell.BasicLSTMCell(HIDDEN_SIZE)
    initial_state = state = tf.zeros([BATCH_SIZE, cell.state_size], dtype = np.float32)

    words = tf.placeholder(tf.float32, [BATCH_SIZE, NUM_STEPS])
    target_words = tf.placeholder(tf.int32, [BATCH_SIZE, NUM_STEPS])

    W_soft = tf.Variable(tf.truncated_normal([HIDDEN_SIZE, num_chars], stddev = 0.01))
    b_soft = tf.Variable(tf.constant(0.01, shape = [num_chars]))

    loss = tf.Variable(tf.constant(0.0))
    predictions = []
    for i in range(0, NUM_STEPS):
        with tf.variable_scope("LSTM" + str(i)):
            output, state = cell(tf.reshape(words[:, i], [BATCH_SIZE, 1]), state)
            prediction = tf.nn.softmax(tf.matmul(output, W_soft) + b_soft)
            predictions.append(prediction)

            for j in range(0, BATCH_SIZE):
                loss = tf.add(loss, tf.log(tf.slice(prediction, tf.pack([j, target_words[j, i]]), [1, 1])))

    loss = - tf.truediv(loss, float(NUM_STEPS) / float(BATCH_SIZE))
    final_state = state

    # define train step
    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

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

    # saving the model
    saver = tf.train.Saver()
    sess.run(tf.initialize_all_variables())
    checkpoint = tf.train.get_checkpoint_state("saved_networks")
    if checkpoint and checkpoint.model_checkpoint_path and behavior == "test":
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print "Successfully loaded:", checkpoint.model_checkpoint_path

        # random seed
        ch = random.randrange(num_chars)
        
        # repeatedly sample text
        buff = ""
        current_state = initial_state.eval()
        for i in range(0, NUM_SEQUENCES):
            gen_words = np.zeros((BATCH_SIZE, NUM_STEPS), dtype = np.int32)
            gen_words[1, 1] = ch
            for j in range(0, NUM_STEPS):
                next_char_dist = predictions[j].eval(feed_dict = {initial_state : current_state,
                                                                  words : gen_words})
                # sample a character
                choice = -1
                point = random.random()
                weight = 0.0
                for p in range(0, num_chars):
                    weight += next_char_dist[1, p]
                    if weight >= point:
                        choice = p
                        break

                buff += chars[choice]
                gen_words[1, j] = choice

            ch = gen_words[1, -1]
            current_state = final_state.eval(feed_dict = {initial_state : current_state,
                                                          words : gen_words})
        print buff

    else:
        print "Training new network weights"

        # iterate over batches
        for e in range(0, EPOCHS):
            current_state = initial_state.eval()
            total_loss = 0.0

            for i in range(0, epoch_size):
                x = data_batched[:, i * NUM_STEPS : (i + 1) * NUM_STEPS]
                y = data_batched[:, i * NUM_STEPS + 1 : (i + 1) * NUM_STEPS + 1]

                train_step.run(feed_dict = {initial_state : current_state,
                                            words : x,
                                            target_words : y})

                current_state, current_loss = sess.run([final_state, loss],
                                                       feed_dict = {initial_state : current_state, words : x, target_words : y})

                total_loss += current_loss[0][0]

            # save weights
            saver.save(sess, "saved_networks/" + filename, global_step = e)

            total_loss /= epoch_size
            print "Average loss per sequence for epoch", e, ": ", total_loss
        
if __name__ == "__main__":
    main()
