import subprocess as sp
import os

FFMPEG_BIN = "ffmpeg" # on Linux and Mac OS
read_dir = '/Users/hes/Dropbox/231N_Project/3_Data/UCF-101'
write_dir = '/Users/hes/Dropbox/231N_Project/3_Data/UCF-101-img'

action_folders = os.listdir(read_dir)
for action_folder in action_folders:
	action_dir = os.path.join(read_dir, action_folder)
	print(action_dir)
	if os.path.isdir(action_dir):
		write_folder = os.path.join(write_dir, action_folder)
		if not os.path.exists(write_folder):
			os.mkdir(write_folder)
		for img in os.listdir(action_dir):
			if img != '.DS_Store':
				video_name = os.path.join(action_dir, img)
				write_name = os.path.join(write_folder, img[:-4])
				sp.call(
				    'ffmpeg -i {} -f image2 {}-%03d.jpg'.format(video_name, write_name),
				    shell=True)