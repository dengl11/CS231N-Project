"""
Suite of Numeric Operations
"""


import numpy as np


def stack_imgs(imgs):
	"""
	stack an array of images on their final axis
	"""
	return np.stack(imgs, axis = -1)