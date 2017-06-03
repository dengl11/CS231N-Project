import tensorflow as tf
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from deep_cnn.deep_cnn_model import deep_CNN_model
from util.data_loader import *

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
	learning_rate = 3e-3
	decay_rate = 0.9  # decay per epoch
	model_name = 'deep_CNN_modified'
	num_epochs = 12
	#################################################################
	train_dir = 'saved_model/{}/'.format(model_name)
	if not os.path.exists(train_dir):
		os.makedirs(train_dir)
	result_dir = 'result_plot/{}/'.format(model_name)
	if not os.path.exists(result_dir):
		os.makedirs(result_dir)
	# full dataset
	dataset = data_loader('data/kitti_full.npy', 'data/kitti_full_metadata.json')
	# # mini dataset
	# dataset = data_loader('data/kitti_mini.npy', 'data/kitti_mini_metadata.json')
	model = deep_CNN_model(learning_rate, model_name, num_epochs, dataset, train_dir, result_dir)
	sess =  tf.Session()
	model = initialize_model(sess, model, train_dir)
	model.train(sess)

if __name__ == '__main__':
	main()
