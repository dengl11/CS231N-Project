import glob
from scipy import misc
import numpy as np

img_src = 'data/kitti_data/'
file_names = sorted(glob.glob(img_src + '/*/*.png', recursive=True))
data = np.concatenate([misc.imread(i)[np.newaxis, ...] for i in file_names], axis=0)
print("Data loaded! Shape: {}".format(data.shape))
save_path = "data/kitti-compressed"
np.save(save_path, data)
print("Saved to:  {}".format(save_path))
		