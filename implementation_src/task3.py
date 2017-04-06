#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# tf_wgan_tdt4173.py: Rudimentary implementation of a wasserstein GAN that
# is to be used as a base for the second programming task in assignment 4.
#
import sys

import sys

import numpy as np
import tensorflow as tf

import helpers

import matplotlib.pyplot as plt

def task3():
    # You might want to alter the learning rate, number of epochs, and batch size
    batch = tf.Variable(0)
    lr = tf.train.exponential_decay(
        1e-4,  # Base learning rate.
        batch,  # Current index into the dataset.
        10000,  # Decay step.
        0.99,  # Decay rate.
        staircase=True)
    nb_epochs = 40000
    batch_size = 64

    # Set to `None` if you do not want to write out images
    path_to_images = './generated_images'

    z_size = 10
    x_size = 28 * 28
    h_size = 128

    # Defined at the top because we need it for initialising weights
    def create_weights(shape):
        # See paper by Xavier Glorot and Yoshua Bengio for more information:
        # "Understanding the difficulty of training deep feedforward neural networks"
        # We employ the Caffe version of the initialiser: 1/(in degree)
        in_dim = shape[0]
        xavier_dev = 1. / tf.sqrt(in_dim / 2.)
        v = tf.random_normal(shape=shape, stddev=xavier_dev)
        return tf.Variable(v)

    def create_w1(shape):
        return tf.Variable(tf.random_normal(shape, stddev=1))

    #
    # Creation of generator and discriminator networks START here
    # Task (a) is to improve the generator and discriminator networks as they
    # currently do not do very much
    #

    # Define weight matrices for the generator
    # Note: Input of the first layer *must* be `z_size` and the output of the
    # *last* layer must be `x_size`
    weights_G = {
        'w1': create_weights((z_size, h_size)),
        'b1': tf.Variable(tf.zeros(shape=[h_size])),
        'w2': create_weights((h_size, x_size)),
        'b2': tf.Variable(tf.zeros(shape=[784])),
    }

    def generator(z, weights):
        z1 = tf.matmul(z, weights['w1'])
        h1 = tf.nn.relu(z1 + weights['b1'])

        z2 = tf.matmul(h1, weights['w2'])
        out = tf.nn.sigmoid(z2 + weights['b2'])
        return out

    # Define weight matrices for the discriminator
    # Note: Input will always be `x_size` and output will always be 1
    weights_D = {
        'w1': create_weights((x_size, h_size)),
        'b1': tf.Variable(tf.zeros(shape=[h_size])),
        'w2': create_weights((h_size, 1)),
        'b2': tf.Variable(tf.zeros(shape=[1]))
    }

    def discriminator(x, weights):
        z1 = tf.matmul(x, weights['w1'])
        h1 = tf.nn.relu(z1 + weights['b1'])

        z2 = tf.matmul(h1, weights['w2'])
        out = z2 + weights['b2']

        return out

    #
    # Creation of generator and discriminator networks END here
    #

    # Weight clipping (default `c` from the WGAN paper)
    c = 0.01
    clipped_D = [w.assign(tf.clip_by_value(w, -c, c)) for w in weights_D.values()]

    # Definition of how Z samples are generated
    z_sampler = lambda nb, dim: np.random.uniform(-1.0, 1.0, size=(nb, dim))

    # Load MNIST
    mnist = helpers.load_mnist_tf('./mnist')

    # Define model entry-points (Z - generator, X - discriminator)
    Z = tf.placeholder(
        tf.float32,
        shape=(None, z_size))
    X = tf.placeholder(
        tf.float32,
        shape=(None, x_size))

    # Define the different components of a GAN
    sample = generator(Z, weights_G)
    fake_hat = discriminator(sample, weights_D)
    real_hat = discriminator(X, weights_D)

    # Define error functions
    error_G = - tf.reduce_mean(fake_hat)
    error_D = tf.reduce_mean(real_hat) - tf.reduce_mean(fake_hat)

    # Specify that we will use RMSProp (one optimiser for each model)
    optimiser_G = tf.train.RMSPropOptimizer(lr).minimize(error_G,
                                                         var_list=weights_G.values())
    optimiser_D = tf.train.RMSPropOptimizer(lr).minimize(-error_D,
                                                         var_list=weights_D.values(),
                                                         global_step=batch)

    # Generate Op that initialises global variables in the graph
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        # Initialise variables and start the session
        sess.run(init)

        if path_to_images:
            helpers.create_dir(path_to_images)

        # Run a set number of epochs (default `n_critic` from the WGAN paper)
        n_critic = 5
        for epoch in range(nb_epochs):
            for _ in range(n_critic):
                # Retrieve a batch from MNIST
                X_batch, _ = mnist.train.next_batch(batch_size)

                # Clip weights and run one step of the optimiser for D
                sess.run(clipped_D)
                sess.run(optimiser_D, feed_dict={Z: z_sampler(batch_size, z_size),
                                                 X: X_batch})

            # Run one step of the optimiser for G
            sess.run(optimiser_G, feed_dict={Z: z_sampler(batch_size, z_size)})

            # Print out some information every nth iteration
            if epoch % 100 == 0:
                # print(sess.run(real_hat, feed_dict={X: X_batch})[0])
                # print("w1:",sess.run(weights_D['w1']))
                err_G = sess.run(error_G, feed_dict={Z: z_sampler(batch_size, z_size)})
                err_D = sess.run(error_D, feed_dict={Z: z_sampler(batch_size, z_size),
                                                     X: X_batch})
                print('Epoch: ', epoch, 'lr ', sess.run(lr))
                print('\t Generator error:\t     {:.7f}'.format(err_G))
                print('\t Discriminator error:\t {:.7f}'.format(err_D))

            # Plot the image generated from 64 different samples to a directory
            if path_to_images and epoch % 1000 == 0:
                samples = sess.run(sample, feed_dict={Z: z_sampler(64, z_size)})

                figure = helpers.plot_samples(samples)
                plt.savefig('{}/{}.png'.format(path_to_images, str(epoch).zfill(10)),
                            bbox_inches='tight')
                plt.close()
