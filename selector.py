import random

########################################################################
# Assuming the dataSource is a 4 dimensional list, top level classes   
# should also be encoded as integer IDs. Other parameters:
# batchSize: how many examples in each batch
# classes: a list contains the classes of interest
# videos: a list contains the indexes for long videos (within the class) of interest
# clips: a list contains the indexes for the clips (within the videos) of interest
# NOTE: for the above 3 lists, if set to None, then all of the possible outcomes 
# are considered. A None list would requirement all of the following list to be None.
# And only the last non-None list is allowed to have more than one elements.
#
# return: A list (with length = batchSize) of the following list:
# [class_id, video_id, clip_id, center_image_path, left_image_path, right_image_path]
# Also assuming the actual elment (leaves) stored within the initial dataSource
# are absolute/relative filenames or other identifiers for each image
########################################################################


class Selector:
    
    def __init__(self, dataSource, batchSize, classes, videos, clips, missingInBetween=1):
        self.dataSource = dataSource
        self.batchSize = batchSize
        self.constraints = [classes, videos, clips]        
        self.missingInBetween = missingInBetween

    def nextBatch(self):
        batch = []
        for _ in range(self.batchSize):
            example = self._sample([], self.dataSource, 0)
            batch.append(example)
        return batch

    def _sample(self, path, dataSource, depth):
        if (depth == len(self.constraints)):
            center_idx = random.randint(self.missingInBetween, len(dataSource) - self.missingInBetween - 1)
            path.append(dataSource[center_idx - self.missingInBetween])
            path.append(dataSource[center_idx + self.missingInBetween])
            path.append(dataSource[center_idx])
            return path
        idx = random.randint(0, len(datasource) - 1) if self.constraints[depth] == None \
                else  random.choice(self.constraints[depth])
        path.append(idx)
        return self._sample(path, dataSource[idx], depth + 1)
        

# simple test case
map = [
        [[[1,2], [3,4]], [[5,6,7], [8]], [[9, 10, 11], [12], [13]]], 
        [[[14, 15]], [[16, 17], [18]]], 
        [[[19, 20]]]
      ]
s = Selector(map, 5, [0, 1], [0], [0], 0)
print(s.nextBatch())
