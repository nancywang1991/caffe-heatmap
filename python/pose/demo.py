import numpy as np
from applyNet import applyNet
import matplotlib
#matplotlib.use('Agg')
# This file uses a FLIC trained model and applies it to a video sequence from Poses in the Wild
#
# Download the model:
#    wget http://tomas.pfister.fi/models/caffe-heatmap-flic.caffemodel -P ../../models/heatmap-flic-fusion/

# Options
opt = {}
opt["visualize"] = True		# Visualise predictions?
opt["useGPU"] = True 			# Run on GPU
opt["dims"] = [256, 256] 		# Input dimensions (needs to match matlab.txt)
opt["numJoints"] = 7 			# Number of joints
opt["layerName"] = 'conv5_fusion' # Output layer name
opt["modelDefFile"] = '/home/wangnxr/Documents/caffe-heatmap/models/heatmap-flic-fusion/matlab.prototxt' # Model definition
opt["modelFile"] = '/home/wangnxr/Documents/caffe_heatmap/snapshots/_iter_34800.caffemodel' # Model weights
#opt["modelFile"] = '/home/wangnxr/Documents/caffe-heatmap/models/heatmap-flic-fusion/caffe-heatmap-flic.caffemodel' # Model weights


# Image input directory
opt["inputDir"] = '/home/wangnxr/Documents/caffe-heatmap/matlab/pose/sample_images/'

# Create image file list
files = {}
imInds = np.arange(1,30)
for ind in xrange(len(imInds)):
    files[ind] = '%05i.jpg' % imInds[ind]

# Apply network
joints = applyNet(files, opt)
