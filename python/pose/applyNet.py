# Wrapper to run network on multiple images
import numpy as np
from applyNetImage import applyNetImage
from initCaffe import initCaffe

def applyNet(files, opt):
    opt["numFiles"] = len(files)
    net = initCaffe(opt)
    joints = np.zeros(shape=(2,opt["numJoints"], opt["numFiles"]))
    for ind in xrange(opt["numFiles"]):
        imFile = files[ind]
        print "file: %s" % imFile
        joints[:,:, ind] = applyNetImage(opt["inputDir"] + imFile, net, opt)
    return joints