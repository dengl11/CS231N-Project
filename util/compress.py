import glob
from scipy import misc
import numpy as np

img_src = 'data/kitti_data/'
file_names = sorted(glob.glob(img_src + '/*/*.png', recursive=True))
count = 0
resized_img_array = []
for i in file_names:
	resized_img_array.append(misc.imresize(misc.imread(i), (128, 384)))
	count += 1
	if count % 1000 == 0:
		print("Processed {} images".format(count))
data = np.concatenate([i[np.newaxis, ...] for i in resized_img_array], axis=0)

print("Data loaded! Shape: {}".format(data.shape))
save_path = "data/kitti-compressed-resized"
np.save(save_path, data)
print("Saved to:  {}".format(save_path))
		