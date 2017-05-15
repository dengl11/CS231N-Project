############################################################
###############  Toolkit for Image Operations  #############
############################################################
import os, sys
from os import walk
from PIL import Image

def imgs_in_folder(folder):
	imgs = [p[2] for p in walk(folder)][0]
	imgs = list(filter(lambda x: not x.startswith("."), imgs))
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
	except: os.mkdir(directory)


def resize(img_path, size, output_folder):
	"""
	resize single image from path, to output_folder
	size: (width, hight)
	"""

	try:
		im = Image.open(img_path)
		im.thumbnail(size, Image.ANTIALIAS)
		im.save(os.path.join(output_folder, img_path.split("/")[-1]), "JPEG")
	except IOError:
		print("cannot create thumbnail for '%s'" % img_path)


def resize_all(input_folder, output_folder, size):
	"""
	resize all images in input folder, and dump to output folder
	size: (width, hight)
	"""
	create_dir(output_folder)
	for m in imgs_in_folder(input_folder):
		resize(os.path.join(input_folder, m), size, output_folder)
 	