import tensorflow as tf
import time, os, sys
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['figure.figsize'] = (20, 20) # set default size of plots
plt.rcParams['image.cmap'] = 'gray'
from util.data_loader import *
from deep_cnn.deep_cnn_model import deep_CNN_model

def initialize_model(session, model, train_dir):
    ckpt = tf.train.get_checkpoint_state(train_dir)
    v2_path = ckpt.model_checkpoint_path + ".index" if ckpt else ""
    if ckpt and (tf.gfile.Exists(ckpt.model_checkpoint_path) or tf.gfile.Exists(v2_path)):
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        print("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
        print('Num params: %d' % sum(v.get_shape().num_elements() for v in tf.trainable_variables()))
    return model

def main():
	######################### Hyperparameter#########################
	learning_rate = 1e-3
	decay_rate = None
	model_name = 'deep_CNN'
	num_epochs = 3
	#################################################################
	train_dir = 'saved_model/' + model_name + "/"
	if not os.path.exists(train_dir):
		os.makedirs(train_dir)
	result_dir = 'result_plot/' + model_name + "/"
	if not os.path.exists(result_dir):
		os.makedirs(result_dir)
	
	# mini dataset
	dataset = data_loader('data/kitti_mini.npy', 'data/kitti_mini_metadata.json')
	# # full dataset
	# dataset = data_loader('data/kitti_full.npy', 'data/kitti_full_metadata.json')
	model = deep_CNN_model()
	if decay_rate is None:
		optimizer = tf.train.AdamOptimizer(learning_rate)
		# batch normalization in tensorflow requires this extra dependency
		extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		with tf.control_dependencies(extra_update_ops):
		    train_step = optimizer.minimize(model.loss)
		    pass

	sess =  tf.Session()
	initialize_model(sess, model, train_dir)
	train_variables = [model.loss, train_step]
	test_variables = [model.loss]
	history = []
	train_loss_history = []
	test_loss_history = []
	best_test_loss = None
	for e in range(num_epochs):
	    batch_num = 0
	    epoch_loss = []
	    epoch_start = time.time()
	    print("Training Epoch {}".format(e+1))
	    pbar = tqdm(dataset.get_minibatches(), total = int(len(dataset.train_x_start_index)/16))
	    for batch_x, batch_y in pbar:
	        loss, _ = sess.run(train_variables,feed_dict={model.X: batch_x, model.y: batch_y})
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
	    for batch_x, batch_y in dataset.get_minibatches(training = False):
	    	loss = sess.run(test_variables,feed_dict={model.X: batch_x, model.y: batch_y})[0]
	    	test_epoch_loss.append(loss)
	    test_loss = sum(test_epoch_loss)/len(test_epoch_loss)
	    test_loss_history.append(test_loss)
	    print("Mean Test Batch Loss: {:.3e}".format(test_loss))
	    if best_test_loss == None or test_loss < best_test_loss:
	    	model.saver.save(sess, train_dir)
	plt.plot(history)
	plt.savefig(result_dir+'loss_curve.png')

if __name__ == '__main__':
  main()
