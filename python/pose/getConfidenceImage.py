import cv2
import numpy as np
import pdb
import matplotlib.pyplot as plt
# Visualize heatmap

def getConfidenceImage(dist, segcpimg_crop, clrs):
    num_points = dist.shape[2]
    m,n,_ = segcpimg_crop.shape
    bbox = [1,1,n,m]
    background = 1-cv2.Canny(cv2.cvtColor(cv2.convertScaleAbs(segcpimg_crop), cv2.COLOR_BGR2GRAY), 100, 200)

    pdf_img = 0.2*np.tile(background, (3,1,1)) + 0.8*(np.zeros(shape=(3,bbox[3], bbox[2]))+1)

    # normalize the distributions for visualization

    for c in range(num_points-1, -1, -1):
        alpha = np.tile(dist[:,:,c]/np.max(dist[:,:,c]), (3,1,1))
        single_joint_pdf = np.ndarray.transpose(np.tile([clrs[c,0], clrs[c,2], clrs[c,1]], (dist.shape[0], dist.shape[1], 1)), (2,0,1))
        pdf_img = np.multiply(alpha, single_joint_pdf) + np.multiply((1-alpha), pdf_img)

    return np.ndarray.transpose(pdf_img, (1,2,0))