import numpy as np

# Find joints in heatmap (== max locations in heatmap)
def heatmapToJoints(heatmapResized, numJoints):
    joints = np.zeros(shape=(2, numJoints))
    for i in xrange(numJoints):
        sub_img = heatmapResized[:, :, i]
        y,x = np.unravel_index(sub_img.argmax(), sub_img.shape)
        joints[:,i] = (x,y)
    return joints