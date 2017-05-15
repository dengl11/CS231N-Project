############################################################
###############  Toolkit for Image Operations  #############
############################################################

def merge(end_points, mid):
	"""
	merge generated middle images with end-point images
	"""
	assert len(end_points) == len(mid) + 1, "len(end_points) == len (mid) + 1 not satisfied!"
	merged = end_points[:]
	for i, img in enumerate(mid): merged.insert(2*i+1, img)
	return merged