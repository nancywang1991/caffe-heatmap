import numpy as np
import pdb
import cv2

# Prepare input image for caffe: change to single & permute color channels
def prepareImagePose(img, transpose=True):

    imgOut = cv2.resize(img, (256,256))

    if transpose:
        imgOut = np.array([imgOut[:, :, 2], imgOut[:, :, 1], imgOut[:, :, 0]])
        imgOut = np.transpose(imgOut, (0,1,2))
        imgOut = np.array([imgOut])

    return imgOut
