############################################################
###############  Toolkit for Image Operations  #############
############################################################
import os, sys
from os import walk
from PIL import Image
from scipy import misc

def files_in_folder(folder, format="jpeg"):
	"""
	return a list of file names in folder
	"""
	imgs = [p[2] for p in walk(folder)][0]
	imgs = list(filter(lambda x:  x.endswith(format), imgs))
	return imgs



def imgs_in_folder(folder, format="jpeg"):
	return [misc.imread(os.path.join(folder, x)) for x in files_in_folder(folder, format=format)]


def merge(end_points, mid):
	"""
	merge generated middle images with end-point images
	"""
	assert len(end_points) == len(mid) + 1, "len(end_points) == len (mid) + 1 not satisfied!"
	merged = end_points[:]
	for i, img in enumerate(mid): merged.insert(2*i+1, img)
	return merged


def create_dir(directory):
	try: os.stat(directory)
	except: 
		print("{} not existing. Just created!".format(directory))
		os.mkdir(directory)


def resize(img_path, size, output_folder, keep_ratio=False):
	"""
	resize single image from path, to output_folder
	size: (width, hight)
	"""
	try:
		im = Image.open(img_path)
		if keep_ratio: im.thumbnail(size, Image.ANTIALIAS)
		else:          im.thumbnail(size)
		im.save(os.path.join(output_folder, img_path.split("/")[-1]), "JPEG")
	except IOError:
		print("cannot create thumbnail for '%s'" % img_path)


def resize_all(input_folder, output_folder, size):
	"""
	resize all images in input folder, and dump to output folder
	size: (width, hight)
	"""
	create_dir(output_folder)
	for m in files_in_folder(input_folder):
		resize(os.path.join(input_folder, m), size, output_folder)


def avg_imges(x1, x2, dtype='uint8'):
    return np.array([x1, x2]).mean(axis=0).astype(dtype)
 	