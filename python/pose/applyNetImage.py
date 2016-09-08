import cv2
import matplotlib.pyplot as plt
import pdb
import numpy as np

from prepareImagePose import prepareImagePose
from processHeatmap import processHeatmap
from getConfidenceImage import getConfidenceImage
from plotSkeleton import plotSkeleton


# Apply network to a single image
def applyNetImage(imgFile, net, opt):
    img = cv2.imread(imgFile)
    input_data = prepareImagePose(img)
    net.blobs['data'].data[...] = input_data
    net.forward()
    features = net.blobs[opt["layerName"]].data
    joints, heatmaps = processHeatmap(features, opt)

    for i in xrange(7):
        plt.imshow(heatmaps[:,:,i])
        plt.show()
    pdb.set_trace()
    if opt["visualize"]:
        visualize(heatmaps, prepareImagePose(img, transpose=False), joints)
    return joints

def visualize(heatmaps, img, joints):
    colours = np.array([np.array([0, 0, 1]),np.array([0, 1, 0]),np.array([1, 0, 0]),np.array([1, 1, 0]),np.array([0, 1, 1]),np.array([1, 0, 1]),np.array([0, 0, 0])])
    clrs = colours
    heatmapVis = getConfidenceImage(heatmaps, img, clrs)
    plt.imshow(heatmapVis)
    plotSkeleton(joints, {}, {}, True)
    plt.show()


