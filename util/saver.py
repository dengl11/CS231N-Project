import numpy as np

def save_learning_curve(iterations, loss, save_path):
	"""
	save data of learning curve to file
	"""
	np.savez_compressed(save_path, iterations = iterations, loss = loss)
	print("Data saved to: {}".format(save_path))