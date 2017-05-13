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
	metadata = {}
	# filter out .DS_Store
	actions = [i for i in os.listdir(data_folder) if i != '.DS_Store']
	for action in actions:
		if action not in metadata:
			metadata[action] = []
		action_dir = os.path.join(data_folder, action)
		files = [i for i in os.listdir(action_dir) if i != '.DS_Store']
		## 'v_ApplyEyeMakeup_g01_c02-116.jpg'
		for file in files:
			_, action_name, video, clip = file[:-4].split('_')
			clip, photo = clip.split('-')
			# -1 for zero indexing
			video = int(video[1:]) - 1
			clip = int(clip[1:]) - 1
			photo = int(photo[1:]) - 1
			if video == len(metadata[action]):
				metadata[action].append([])
			video_list = metadata[action][video]
			if clip == len(video_list):
				metadata[action][video].append([])
			clip_list = metadata[action][video][clip]
			if photo == len(clip_list):
				full_path = os.path.join(action_dir, file)
				metadata[action][video][clip].append(full_path)
	return metadata

os.chdir('..')
cwd = os.getcwd()
full_data_fld = os.path.join(cwd, 'data','UCF-101-img')
test_data_fld = os.path.join(cwd, 'data','UCF-101-img-test')
full_metadata = create_metadata(full_data_fld)
test_metadata = create_metadata(test_data_fld)
with open('data/test_metadata.json', 'w') as outfile:
    json.dump(test_metadata, outfile)
			
with open('data/full_metadata.json', 'w') as outfile:
    json.dump(full_metadata, outfile)
