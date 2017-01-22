import cv2
import matplotlib
#matplotlib.use("agg")
import matplotlib.pyplot as plt
import pdb
import numpy as np

from prepareImagePose import prepareImagePose
from processHeatmap import processHeatmap
from getConfidenceImage import getConfidenceImage
from plotSkeleton import plotSkeleton


# Apply network to a single image
def applyNetImage(img, net, opt):
    input_data = prepareImagePose(img)
    net.blobs['data'].data[...] = input_data
    net.forward()
    features = net.blobs[opt["layerName"]].data
    heatmaps = processHeatmap(features, opt)

    #for i in xrange(7):
    #    plt.imshow(heatmaps[:,:,i])
    #    plt.show()
    #pdb.set_trace()
    if opt["visualize"]:
        visualize(heatmaps, prepareImagePose(img, transpose=False), joints)
    return joints

def visualize(heatmaps, img, joints):
    colours = np.array([np.array([0, 0, 1]),np.array([0, 1, 0]),np.array([1, 0, 0]),np.array([1, 1, 0]),np.array([0, 1, 1]),np.array([1, 0, 1]),np.array([0, 0, 0])])
    clrs = colours

    heatmapVis, background = getConfidenceImage(heatmaps, img, clrs)
    fig, ax = plt.subplots()
    ax.imshow(heatmapVis)
    #ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    #plt.imshow(np.ndarray.transpose(background))
    plotSkeleton(joints, confidence, {}, {}, ax, True)
    plt.show()



