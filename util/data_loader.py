import numpy as np
from scipy import misc
import json, time


class data_loader(object):
	def __init__(self, data_src, metadata_src, gap = 1):
		## loading data file and metadata
		time1 = time.time()
		self.data_src = data_src
		self.metadata_src = metadata_src
		self.gap = gap
		self.data = np.load(data_src)
		with open(metadata_src) as metadata_file:
			self.metadata = json.load(metadata_file)
		self.video_names = sorted(list(self.metadata.keys()))
		self.train_test_split()
		time2 = time.time()
		print("Data loaded! Took {:.2f} seconds".format(time2 - time1))
		print("Data Shape {}".format(self.data.shape))
		print("train: {0}   test: {1}".format(len(self.train_x_start_index), len(self.test_x_start_index)))
	
	def train_test_split(self, train_percent = 0.8):
		# 80% of the video as train and 20% test
		train_index = []
		test_index = []
		train_video = list(range(int(train_percent * len(self.metadata))))
		test_video = list(range(int(train_percent * len(self.metadata)), len(self.metadata)))
		# training set
		self.train_x_start_index, self.train_x_end_index, self.train_y_index = self.generate_index(train_video)
		# test set
		self.test_x_start_index, self.test_x_end_index, self.test_y_index = self.generate_index(test_video)


	def generate_index(self, clip_indices):
	    x_start_index = []
	    x_end_index = []
	    y_index = []
	    mid_gap = self.gap //2
	    for i in clip_indices:
	        clip_list = self.metadata[self.video_names[i]]
	        start_index = list(range(0, len(clip_list) - self.gap))
	        end_index = [i + self.gap for i in start_index]
	        mid_index = [i + mid_gap for i in start_index]
	        x_start_index += [self.metadata[self.video_names[i]][j] for j in start_index]
	        x_end_index += [self.metadata[self.video_names[i]][j] for j in end_index]
	        y_index += [self.metadata[self.video_names[i]][j] for j in mid_index]
	    return (x_start_index, x_end_index, y_index)

	def get_batch(self, batch_size = 8, training = True):
		if training:
			rand_incies = np.random.choice(self.train_x_start_index, size = batch_size)
			start_frames = self.data[rand_incies, :, :, :]
			end_frames = self.data[rand_incies, :, :, :]
		else:
			rand_incies = np.random.choice(self.test_x_start_index, size = batch_size)
			start_frames = self.data[rand_incies, :, :, :]
			end_frames = self.data[rand_incies, :, :, :]
		X_batch = np.concatenate([start_frames, end_frames], axis = 3)
		y_batch = self.data[rand_incies, :, :, :]
		return(X_batch, y_batch)



		