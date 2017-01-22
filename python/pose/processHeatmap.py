import cv2
import numpy as np
from heatmapToJoints import heatmapToJoints
import pdb


# Reformat output heatmap: rotate & permute color channels
def processHeatmap(heatmap, opt):

    numJoints = opt["numJoints"]
    heatmapResized = []
    for j in xrange(numJoints):
        heatmapResized.append(cv2.resize(heatmap[0,j,:,:], (opt["dims"][1], opt["dims"][0])))
    heatmapResized = np.ndarray.transpose(np.array(heatmapResized)-1, (1,2,0)).clip(min=0)
   
    return heatmapResized
