# Wrapper to run network on multiple images
import matplotlib.pyplot as plt
import numpy as np
from applyNetImage import applyNetImage
from initCaffe import initCaffe
from plotSkeleton import plotSkeleton
import skvideo.io

def applyNet(vid, opt):
    opt["numFiles"] = len(vid)
    net = initCaffe(opt)
    heatmaps = np.zeros(shape=(opt["numFiles"], vid.shape[1], vid.shape[2]))
    for ind in xrange(opt["numFiles"]):
        image = vid[ind,:,:]
        print "frame: %s" % ind
        heatmaps[ind,:,:] = applyNetImage(image, net, opt)
    return heatmaps

def save_visualization(vid, joints, savename):
    colours = np.array([np.array([0, 0, 1]),np.array([0, 1, 0]),np.array([1, 0, 0]),np.array([1, 1, 0]),np.array([0, 1, 1]),np.array([1, 0, 1]),np.array([0, 0, 0])])
    clrs = colours
    outvid = []
    for frame in vid:
        plt.imshow(frame)
        plotSkeleton(joints, {}, {}, True)
        plt.canvas.draw()
        data = np.fromstring(plt.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(plt.canvas.get_width_height()[::-1] + (3,))
        outvid.append(data)
        plt.clf()
    skvideo.io.vwrite(np.array(outvid), savename)