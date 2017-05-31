############################################################
###############  Toolkit for Image Operations  #############
############################################################
import os, sys
from os import walk
from PIL import Image
from scipy import misc
import numpy as np
from scipy.misc import *
from itertools import product

def files_in_folder(folder, format="jpeg"):
	"""
	return a list of file names in folder
	"""
	try: os.stat(folder)
	except: 
		print("{} is not a valid path!".format(folder))
		return
	imgs = [p[2] for p in walk(folder)][0]
	imgs = list(filter(lambda x:  x.endswith(format), imgs))
	return imgs

def flipped_frames(frames):
    return frames[::-1]


def filter_files(files):
	return list(filter(lambda x: not x.startswith('.'), files))

def split_train_dev(data, train_ratio = 0.8):
    """
    Input:
        data: np array
    Return:
        train_ind: index of training data
        val_ind:   index of validation data
    """
    n = data.shape[0]
    num_train = int(n*train_ratio)
    train_ind = np.random.choice(range(n), num_train, replace=False)
    train_ind.sort()
    val_ind = list(set(range(n)) - set(train_ind))
    val_ind.sort()
    return train_ind, val_ind


def load_imgs(file):
    data = np.load(file)
    imgs, info = data['imgs'], data['info']
    print(info)
    return imgs

def add_background_to_img(img, background):
	return img + background

def add_background_to_imgs(imgs, background):
	return [add_background_to_img(imgs, b) for b in background]

def augment_add_background(collections):
	background = load_imgs("data/moving-box/processed/background.npz")
	ans = collections[:]
	for x in collections:
		ans += add_background_to_imgs(x, background)
	return ans

def augment_data(collections):
	collections = augment_reverse_sequence(collections)
	collections = augment_add_background(collections)
	# collections = augment_reverse_color(collections)
	return collections
	

def augment_reverse_sequence(collections):
	return collections + [x[::-1] for x in collections]
	

def reverse_color(imgs):
	"""
	Input:
		imgs: np array between [0, 1]
	Output:
		np array between [0, 1]
	"""
	return 1-imgs

def augment_reverse_color(collections):
	return collections + [reverse_color(x) for x in collections]


def center_imgs(imgs):
	"""
	Input:
		imgs: np array between [0, 1]
	Output:
		np array between [-1, 1]
	"""
	return 2*imgs - 1

def center_collections(collections):
	return [center_imgs(x) for x in collections]

def get_collection(folder, augment=False):
	data_collection = [p[2] for p in walk(folder)][0]
	data_collection = filter_files(data_collection)
	img_collections = [load_imgs(os.path.join(folder, f)) for f in data_collection]
	if augment: img_collections = augment_data(img_collections)
	return img_collections


def get_processed_moving_box(augment = False):
    return get_collection("data/moving-box/processed", augment)
    

def get_processed_moving_box_squares(augment = False):
	return get_collection("data/moving-box/processed/Box", augment)

def get_processed_diamond(augment = False):
	return get_collection("data/moving-box/processed/diamond", augment)

def get_processed_rectangle(augment = False):
	return get_collection("data/moving-box/processed/rectangle", augment)

def get_processed_cirlce(augment = False):
	return get_collection("data/moving-box/processed/circle", augment)


def rgb2gray(rgb):
	"""
	Dimension:  [H, W, 3] -> [H, W]
	Type:       uint8 [0, 255] -> float32 [0, 1]
	"""
	return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])/255

def imgs_in_folder(folder, format="jpeg"):
	imgs = [misc.imread(os.path.join(folder, x)) for x in files_in_folder(folder, format=format)]
	assert len(imgs)>0, "No images in folder!"
	return imgs


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


def resize(img, size):
	"""
	resize image to target size
	INPUT: 
		img: numpy array of source image
	"""
	try:
		return imresize(img, size)
	except IOError:
		print("cannot create thumbnail for '%s'" % img_path)


def resize_and_save(img_path, size, output_folder, keep_ratio=False):
	"""
	resize single image from path, to output_folder

	INPUT: 
		img_path:   	path of image file on disk
		size: 			(width, hight)
	OUTPUT:
		output_folder:  path of folder to be saved
	"""
	im = Image.open(img_path)
	img.thumbnail(size)
	resized = resize(img_path, size)
	im.save(os.path.join(output_folder, img_path.split("/")[-1]), "JPEG")


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


def scale_loss(loss, init_range): 
	return loss * 255 / init_range
 	