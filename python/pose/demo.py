import numpy as np
from applyNet import applyNet, applyNet_im
import skvideo.io
import glob
from applyNet import save_visualization, save_joint_values, load_joint_file
from flow import calc_flow_video
import pdb
import os
import shutil
import getpass
import subprocess
# This file uses a FLIC trained model and applies it to a video sequ\ence from Poses in the Wild
#
# Download the model:
#    wget http://tomas.pfister.fi/models/caffe-heatmap-flic.caffemodel -P ../../models/heatmap-flic-fusion/

# Options
password=getpass.getpass()

for iteration in [70000]:
    opt = {}
    opt["gpu_id"] = 1
    opt["itr"] = iteration
    opt["visualize"] = False		# Visualise predictions?
    opt["useGPU"] = True 			# Run on GPU
    opt["dims"] = [256, 256] 		# Input dimensions (needs to match matlab.txt)
    opt["numJoints"] = 7 			# Number of joints
    opt["layerName"] = 'conv5_fusion' # Output layer name
    opt["modelDefFile"] = '/home/wangnxr/Documents/caffe-heatmap/models/heatmap-flic-fusion/matlab.prototxt' # Model definition
    opt["modelFile"] = '/home/wangnxr/Documents/caffe-heatmap/models/_iter_%i.caffemodel' % iteration # Model weights
    #opt["modelFile"] = '/home/wangnxr/Documents/caffe_heatmap/_iter_55600.caffemodel' # Model weights

    opt["saveDir"] = '/mnt/pose_results/cb46fd46/' # Model weights
    opt["floDir"] = '/mnt/flo_files/'
    #opt["floDir"] = '/media/wangnxr/b1d81c2f-943e-421f-b6bc-75e9e33bac6c/results/flo_files/new_patients/'
    opt["use_flow"] = False
    opt["type"] = "vid"
    opt["skeleton"] = True
    opt["blur_face"]= True
    opt["orig_size"] = False
    opt["warp_all"] = False
    opt["use_prev"] = True
    opt["alt_video"] = False
    opt["redo_old"] = False
    #opt["modelFile"] = '/home/wangnxr/Documents/caffe-heatmap/models/heatmap-flic-fusion/caffe-heatmap-flic.caffemodel' # Model weights


    # Video input directory
    opt["inputDir_alt"] = '/mnt/data/cb46fd46_5/'

    opt["inputDir"] = '/mnt/results/cb46fd46/cb43fd46_7/'
    #opt["vid_nums"] = [736]
    #opt["vid_nums"] = [37, 77, 101, 104, 167, 217, 245, 271, 289, 327, 347, 364, 464, 516, 551, 663, 674]
    #opt["inputDir"] = '/home/wangnxr/Documents/video_data/whole_patient_preprocessed/cb46fd46_4/special/'
    #opt["vid_nums"] = [0]
    # Create image file list
    files = {}
    if opt["type"] == "img":
        for img_name in sorted(glob.glob(opt["inputDir"] + "/30*.png")):
            print "Loading image: %s" % img_name
            vid_name = img_name.split("/")[-1].split('.')[0]
            # Apply network
            joints, heatmaps = applyNet_im(img_name, opt)
    else:
       # pdb.set_trace()
        if "vid_nums" in opt:
            vid_fnames = sorted(glob.glob(opt["inputDir"] + "/*%04i.mp4.enc" % num)[0] for num in opt["vid_nums"])
        else:
            vid_fnames = sorted(glob.glob(opt["inputDir"] + "/*.mp4.enc"))
        for vid_fname in vid_fnames:
            if opt["redo_old"] or not os.path.exists(opt["saveDir"]+vid_fname.split("/")[-1].split(".")[0]+".avi.enc"):
                
                print "Loading video: %s" % vid_fname
                subprocess("openssl enc -d -des -in %s -out %s -pass pass:%s" % (vid_fname, vid_fname[:-4], password), shell=True)
                vid = skvideo.io.vread(vid_fname[:-4])
                print "Loaded"
                vid_name = vid_fname.split("/")[-1].split('.')[0]
                joint_file = opt["saveDir"] + vid_fname.split("/")[-1].split(".")[0] + ".txt"
                if opt["use_prev"] and os.path.exists(joint_file):
                    print "Using previous joint file %s" % joint_file
                    joints, confidences = load_joint_file(joint_file)
                #pdb.set_trace()
                else:
                # Apply network
                    #pdb.set_trace()
                    joints, confidences, heatmaps = applyNet(vid, opt)
                    print "Heatmap done."
                    if opt["use_flow"]:
                        joints, confidences = calc_flow_video(vid_name, heatmaps, opt)
                        print "Optical flow done."
                    save_joint_values(joints, confidences, joint_file)
                            
                if opt["alt_video"]:
                    vid = skvideo.io.vread(opt["inputDir_alt"] + vid_fname.split("/")[-1].split(".")[0] + ".avi")
                #save_visualization(vid, joints, confidences, opt["saveDir"]  + vid_fname.split("/")[-1], opt)
            

