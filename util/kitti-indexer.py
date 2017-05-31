import os, json

#### Folder Structure #####
# CS231N-Project/
# 	util/
# 		loader.py
# 		selector.py
# 	data/
# 		UCF-101-img
# 		UCF-101-img-test
# 		full_metadata.json (CREATE BY THIS SCRIPT)
# 		test_metadata.json (CREATE BY THIS SCRIPT)
###########################
def create_metadata(data_folder):
	count = -1
	metadata = {}
	# filter out .DS_Store
	videos = sorted([i for i in os.listdir(data_folder) if i != '.DS_Store'])
	for video in videos:
		if video not in metadata:
			metadata[video] = []
		video_dir = os.path.join(data_folder, video)
		files = [i for i in os.listdir(video_dir) if i != '.DS_Store']
		## 'v_ApplyEyeMakeup_g01_c02-116.jpg'
		for file in files:
			count += 1
			name, extension = file.split('.')
			metadata[video].append(count)
			
	return metadata

os.chdir('..')
cwd = os.getcwd()
data_fld = os.path.join(cwd, 'data','kitti_data')
metadata = create_metadata(data_fld)
with open('data/kitti_full_metadata.json', 'w') as outfile:
    json.dump(metadata, outfile)
