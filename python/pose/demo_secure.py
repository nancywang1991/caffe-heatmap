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
import argparse
# This file uses a FLIC + Patient video trained model and applies it to a user selected video directory
#
# Download the model:
#    wget http://tomas.pfister.fi/models/caffe-heatmap-flic.caffemodel -P ../../models/heatmap-flic-fusion/


def main(args, password):
    opt = {}
    #Constants
    opt["visualize"] = False		# Visualise predictions?
    opt["useGPU"] = True 			# Run on GPU
    opt["dims"] = [256, 256] 		# Input dimensions (needs to match matlab.txt)
    opt["numJoints"] = 7 			# Number of joints
    opt["layerName"] = 'conv5_fusion' # Output layer name

    #Primary Options
    opt["gpu_id"] = args.gpu_id
    opt["modelDefFile"] = args.model_def#'/home/wangnxr/Documents/caffe-heatmap/models/heatmap-flic-fusion/matlab.prototxt' # Model definition
    opt["modelFile"] = args.model_weight #'/home/wangnxr/Documents/caffe-heatmap/models/_iter_70000.caffemodel' # Model weights
    opt["saveDir"] = args.save_dir #'/mnt/pose_results/cb46fd46/' # Model weights
    opt["floDir"] = args.flo_save_dir #'/mnt/flo_files/'

    opt["inputDir_alt"] = args.vid_alt#'/mnt/data/cb46fd46_5/' # Use these videos if alt_video is True
    opt["inputDir"] = args.vid#'/mnt/results/cb46fd46/cb46fd46_8/' # Primary Video input directory

    #Secondary Options
    opt["use_flow"] = True
    opt["skeleton"] = True
    opt["blur_face"]= True
    opt["orig_size"] = False
    opt["warp_all"] = False
    opt["use_prev"] = True
    opt["alt_video"] = False
    opt["redo_old"] = False
    opt["save_vis"] = False

    if not args.vid_nums is None:
        vid_fnames = sorted(glob.glob(opt["inputDir"] + "/*%04i.mp4.enc" % num)[0] for num in args.vid_nums)
    else:
        vid_fnames = sorted(glob.glob(opt["inputDir"] + "/*.mp4.enc"))
    for vid_fname in vid_fnames:
        if opt["redo_old"] or not os.path.exists(opt["saveDir"]+vid_fname.split("/")[-1].split(".")[0]+".avi.enc"):
            print "Loading video: %s" % vid_fname
            #pdb.set_trace()
            subprocess.call("openssl enc -d -des -in %s -out %s -pass pass:%s" % (vid_fname, vid_fname[:-4], password), shell=True)
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
                os.remove(vid_fname[:-4])
            if opt["save_vis"]:
                if opt["alt_video"]:
                    vid = skvideo.io.vread(opt["inputDir_alt"] + vid_fname.split("/")[-1].split(".")[0] + ".avi")
                save_visualization(vid, joints, confidences, opt["saveDir"]  + vid_fname.split("/")[-1], opt)
if __name__== "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--vid', required=True, help="Video directory")
    parser.add_argument('-va', '--vid_alt', help="Alternate video directory")
    parser.add_argument('-s', '--save_dir', required=True, help="Save Directory" )
    parser.add_argument('-md', '--model_def',
                        default='/home/wangnxr/Documents/caffe-heatmap/models/heatmap-flic-fusion/matlab.prototxt',
                        help='Model definitions' )
    parser.add_argument('-mw', '--model_weight',
                        default = '/home/wangnxr/Documents/caffe-heatmap/models/_iter_70000.caffemodel',
                        help = 'trained model weights')
    parser.add_argument('-f', '--flo_save_dir',
                        default = '/mnt/flo_files/',
                        help = 'location to save temporary flo files')
    parser.add_argument('-gpu', '--gpu_id', default=0, type=int, help = "Which gpu to use")
    parser.add_argument('-vn', '--vid_nums', nargs= '+', help= "If instantiated, only these videos will be processed")
    parser.add_argument('-pass', '--password', help="password for secure processing")
    args = parser.parse_args()
    if not hasattr(args, "password"):
        password = getpass.getpass()
    else:
	password = args.password
    main(args, password)


