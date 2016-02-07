#!/usr/bin/env python

from tensorflow.models.rnn import rnn_cell
import tensorflow as tf
import sys
import math
import numpy as np
import random
import time

# hyperparameters
BATCH_SIZE = 20
HIDDEN_SIZE = 150
EPOCHS = 100
NUM_STEPS = 200
LEN_GEN = 2000
TEMPERATURE = 0.1
LAYERS = 2

if len(sys.argv) == 3:
    _, filename, behavior = sys.argv
    if behavior == "test":
        BATCH_SIZE = 1
        NUM_STEPS = 1
        
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
    stacked_cell = rnn_cell.MultiRNNCell([cell] * LAYERS)
    initial_state = state = stacked_cell.zero_state(BATCH_SIZE, tf.float32)

    words = tf.placeholder(tf.float32, [BATCH_SIZE, NUM_STEPS, num_chars])
    target_words = tf.placeholder(tf.float32, [BATCH_SIZE, NUM_STEPS, num_chars])

    W_soft = tf.Variable(tf.truncated_normal([HIDDEN_SIZE, num_chars], stddev = 0.01))
    b_soft = tf.Variable(tf.constant(0.005, shape = [num_chars]))

    loss = tf.Variable(tf.constant(0.0))
    with tf.variable_scope("RNN"):
        for i in range(0, NUM_STEPS):
            if i > 0:
                tf.get_variable_scope().reuse_variables()

            output, state = stacked_cell(tf.reshape(words[:, i,:], [BATCH_SIZE, num_chars]), state)
            prediction = tf.nn.softmax(tf.matmul(output, W_soft) + b_soft)

            loss = tf.add(loss, tf.reduce_sum(tf.log(tf.reduce_sum(tf.mul(prediction, target_words[:,i,:]), 1))))

    loss = - tf.truediv(loss, float(BATCH_SIZE))
    final_state = state
    final_prediction = prediction

    # define train step
    train_step = tf.train.AdamOptimizer(0.01).minimize(loss)

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
        seed = "lightsaber "

        current_state = initial_state.eval()

        for ch in seed:
            gen_word = np.zeros((1,1, num_chars), dtype = np.int32)
            gen_word[0, 0, char_idx[ch]] = 1
            current_state = final_state.eval(feed_dict = {initial_state : current_state,
                                                           words : gen_word})

        # repeatedly sample text
        prev_char = seed[-1]
        for i in range(0, LEN_GEN):
            gen_word = np.zeros((1,1, num_chars), dtype = np.int32)
            gen_word[0, 0, char_idx[prev_char]] = 1

            next_char_dist, current_state = sess.run([final_prediction, final_state],
                                                     feed_dict = {initial_state : current_state,
                                                                  words : gen_word})

            next_char_dist = np.array(next_char_dist[0], dtype = np.float32)

            # scale the distribution
            next_char_dist /= TEMPERATURE
            next_char_dist = np.exp(next_char_dist)
            next_char_dist /= sum(next_char_dist)

            # sample a character
            choice = -1
            point = random.random()
            weight = 0.0
            for p in range(0, num_chars):
                weight += next_char_dist[p]
                if weight >= point:
                    choice = p
                    break

            prev_char = chars[choice]
            seed += prev_char

        print seed
        
    else:
        print "Training new network weights"

        # iterate over batches
        current_milli_time = lambda: int(round(time.time() * 1000))
        old_time = current_milli_time()
        for e in range(0, EPOCHS):
            current_state = initial_state.eval()
            total_loss = 0.0

            for i in range(0, epoch_size):
                x = data_batched[:, i * NUM_STEPS : (i + 1) * NUM_STEPS] # Returns words (BATCH_SIZE * NUM_STEPS)
                y = data_batched[:, i * NUM_STEPS + 1 : (i + 1) * NUM_STEPS + 1]

                ohx = np.zeros((BATCH_SIZE, NUM_STEPS, num_chars))
                ohy = np.zeros((BATCH_SIZE, NUM_STEPS, num_chars))
                for j in range(0, BATCH_SIZE):
                    for k in range(0, NUM_STEPS):
                        ohx[j,k,x[j,k]] = 1
                        ohy[j,k,y[j,k]] = 1

                current_state, current_loss, _ = sess.run([final_state, loss, train_step],
                                                       feed_dict = {initial_state : current_state, words : ohx, target_words : ohy})

                total_loss += current_loss

            # save weights
            saver.save(sess, "saved_networks/" + filename, global_step = e)

            print "Per word perplexity for epoch", e, ": ", total_loss / (NUM_STEPS * epoch_size)
            print "Epoch finished in", current_milli_time() - old_time, "milliseconds"
            old_time = current_milli_time()
        
if __name__ == "__main__":
    main()
