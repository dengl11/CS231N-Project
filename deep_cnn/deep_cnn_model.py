import tensorflow as tf
import time, os, sys
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['figure.figsize'] = (20, 20) # set default size of plots
plt.rcParams['image.cmap'] = 'gray'


def setup_model(X,y):
	    conv_block1 = conv_block(X, 48, (3,3))
	    conv_block2 = conv_block(conv_block1, 48, (3,3))
	    conv_block3 = conv_block(conv_block2, 96, (3,3))
	    conv_block4 = conv_block(conv_block3, 96, (3,3))
	    conv_block5 = conv_block(conv_block4, 96, (3,3))
	    dconv_block5 = dconv_block(conv_block5, 96, (4,4), (2,2))
	    dconv_block4 = dconv_block(dconv_block5, 96, (4,4), (2,2), conv_block4)
	    dconv_block3 = dconv_block(dconv_block4, 96, (4,4), (2,2), conv_block3)
	    dconv_block2 = dconv_block(dconv_block3, 48, (4,4), (2,2), conv_block2)
	    dconv_block1 = dconv_block(dconv_block2, 3, (4,4), (2,2))
	    y_out = dconv_block1
	    return y_out

def conv_block(inputs, conv_filter, conv_kernel):
    conv1 = tf.layers.conv2d(inputs, 
                             conv_filter, 
                             conv_kernel,
                             padding='same', 
                             activation=parametric_relu)
    conv2 = tf.layers.conv2d(conv1, 
                             conv_filter, 
                             conv_kernel,
                             padding='same', 
                             activation=parametric_relu)
    conv3 = tf.layers.conv2d(conv2, 
                             conv_filter, 
                             conv_kernel,
                             padding='same', 
                             activation=parametric_relu)
    pooled = tf.layers.max_pooling2d(conv3, (2,2), (2,2))
    return pooled

def dconv_block(inputs, dconv_filter, dconv_kernel, dconv_strides, conv_input = None):
    if conv_input is not None:
        inputs = tf.concat([inputs, conv_input], 3)
    dconv = tf.layers.conv2d_transpose(inputs,
                                       dconv_filter,
                                       dconv_kernel,
                                       dconv_strides,
                                       padding='same',
                                       activation=parametric_relu)
    conv1 = tf.layers.conv2d(dconv, 
                             dconv_filter, 
                             (3,3),
                             padding='same', 
                             activation=parametric_relu)
    conv2 = tf.layers.conv2d(conv1, 
                             dconv_filter, 
                             (3,3),
                             padding='same', 
                             activation=parametric_relu)
    return conv2

def parametric_relu(x):
    with tf.variable_scope("PReLu") as scope:
        alpha = tf.get_variable('alpha', [1],
                           initializer=tf.constant_initializer(0.0),
                            dtype=tf.float32)
        h = tf.maximum(alpha * tf.ones_like(x), x)
    return h


class deep_CNN_model(object):
	def __init__(self, learning_rate, model_name, num_epochs, dataset, train_dir, decay_rate = None):
		epsilon = 0.1
		tf.reset_default_graph()
		self.train_dir = train_dir
		self.X = tf.placeholder(tf.float32, [None, 128, 384, 6])
		self.y = tf.placeholder(tf.float32, [None, 128, 384, 3])
		self.y_out = setup_model(self.X, self.y)
		self.learning_rate = learning_rate
		self.model_name = model_name
		self.num_epochs = num_epochs
		self.dataset = dataset
		# Charbonnier Loss
		epsilon = 0.1
		self.loss = tf.reduce_sum(tf.sqrt((self.y_out - self.y) ** 2 + epsilon ** 2))
		## Optimizer
		if decay_rate is None:
			self.optimizer = tf.train.AdamOptimizer(learning_rate)
			# batch normalization in tensorflow requires this extra dependency
			extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
			with tf.control_dependencies(extra_update_ops):
				self.train_step = self.optimizer.minimize(self.loss)
				pass
		self.saver = tf.train.Saver()
		

	def train(self, sess):
		train_variables = [self.loss, self.train_step]
		test_variables = [self.loss]
		history = []
		train_loss_history = []
		test_loss_history = []
		best_test_loss = None
		################## training loop ##################
		for e in range(self.num_epochs):
			batch_num = 0
			epoch_loss = []
			epoch_start = time.time()
			print("Training Epoch {}".format(e+1))
			pbar = tqdm(self.dataset.get_minibatches(), total = int(len(self.dataset.train_x_start_index)/16))
			for batch_x, batch_y in pbar:
				loss, _ = sess.run(train_variables,feed_dict={self.X: batch_x, self.y: batch_y})
				pbar.set_postfix(loss = "{:.3e}".format(loss))
				pbar.update()
				batch_num += 1
				history.append(loss)
				epoch_loss.append(loss)
			epoch_end = time.time()
			train_loss = sum(epoch_loss)/len(epoch_loss)
			train_loss_history.append(train_loss)
			print('epoch: {0} Mean Loss {1:.3e} Time: {2:.1f}seconds'.format(e+1, train_loss, epoch_end - epoch_start))
			print("Validating with Test set")
			test_epoch_loss = []
			for batch_x, batch_y in self.dataset.get_minibatches(training = False):
				loss = sess.run(test_variables,feed_dict={self.X: batch_x, self.y: batch_y})[0]
				test_epoch_loss.append(loss)
			test_loss = sum(test_epoch_loss)/len(test_epoch_loss)
			test_loss_history.append(test_loss)
			print("Mean Test Batch Loss: {:.3e}".format(test_loss))
			if best_test_loss is None or test_loss < best_test_loss:
				print("New best dev score! Saving model in {}".format(self.train_dir))
				self.saver.save(sess, self.train_dir + self.model_name)

