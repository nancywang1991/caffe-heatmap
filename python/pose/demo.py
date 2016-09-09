import numpy as np
from applyNet import applyNet
import skvideo.io
import glob
from applyNet import save_visualization
from flow import calc_flow_video

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
opt["modelFile"] = '/home/wangnxr/Documents/caffe_heatmap/snapshots/_iter_400.caffemodel' # Model weights
opt["saveDir"] = '/home/wangnxr/Documents/caffe_heatmap/results/' # Model weights
#opt["modelFile"] = '/home/wangnxr/Documents/caffe-heatmap/models/heatmap-flic-fusion/caffe-heatmap-flic.caffemodel' # Model weights


# Video input directory
opt["inputDir"] = '/home/wangnxr/Documents/caffe-heatmap/matlab/pose/sample_images/'

# Create image file list
files = {}

for vid_fname in sorted(glob.glob(opt["inputDir"])):
    vid = skvideo.io.vreader(vid_fname)
    # Apply network
    heatmaps = applyNet(vid, opt)
    joints_list = calc_flow_video(vid, heatmaps, opt)
    save_visualization(vid, joints_list, opt["saveDir"] + vid_fname.split("/")[-1])

