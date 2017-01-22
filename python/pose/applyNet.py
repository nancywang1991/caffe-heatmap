# Wrapper to run network on multiple images
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np
from applyNetImage import applyNetImage
from initCaffe import initCaffe
from plotSkeleton import plotSkeleton
import skvideo.io
import pdb
import cv2


def applyNet(vid, opt):
    opt["numFiles"] = vid.shape[0]
    net = initCaffe(opt)
    heatmaps = np.zeros(shape=(opt["numFiles"], opt["dims"][0], opt["dims"][1], opt["numJoints"]))
    for ind in xrange(opt["numFiles"]):
        image = vid[ind,:,:]
        if ind%100==0:
            print "frame: %s" % ind
        heatmaps[ind,:,:] = applyNetImage(image, net, opt)
    return heatmaps

def applyNet_im(im, opt):

    net = initCaffe(opt)
    image = cv2.imread(im)
    joint, confidence, heatmap = applyNetImage(image, net, opt)

    return joints, confidence, heatmap

def save_visualization(vid, joints, confidences, savename, opt):
    writer = skvideo.io.FFmpegWriter(savename.split('.')[0]+'.avi', inputdict={"-r": "30"}, outputdict={"-vcodec": 'mpeg4', "-qscale:v":"3"})
    crop_coords = open("%s/%s.txt" %(opt["inputDir"], savename.split('/')[-1].split('.')[0])).readlines()
    for f, joint in enumerate(joints):
        if opt["orig_size"]:
            frame = vid[f]
            joint[0,:] *= (vid[f].shape[1]/float(opt["dims"][1]))
            joint[1,:] *= (vid[f].shape[0]/float(opt["dims"][0]))
        elif opt["alt_video"]:
            frame = vid[f]
            crop_coord = [int(coord) for coord in crop_coords[f][:-1].split(",")]
            if sum(crop_coord) <= 0:
                rescale_factor = (vid[f].shape[1]/float(opt["dims"][0]),vid[f].shape[0]/float(opt["dims"][1]))
            else:
                rescale_factor = ((crop_coord[1]-crop_coord[0])/float(opt["dims"][0]), (crop_coord[3]-crop_coord[2])/float(opt["dims"][1]))
            joint[0,:] = joint[0,:]*rescale_factor[0] + crop_coord[0]
            joint[1,:] = joint[1,:]*rescale_factor[1] + crop_coord[2]
            #pdb.set_trace
        else:
            frame = cv2.resize(vid[f], (opt["dims"][0], opt["dims"][1]))
        if opt["blur_face"]:
            x1 = int(joint[0,0]-40)
            y1 = int(joint[1,0]-40)
            x2 = int(joint[0,0]+40)
            y2 = int(joint[1,0]+40)
            #pdb.set_trace()
            frame[y1:y2, x1:x2] = cv2.blur(frame[y1:y2, x1:x2],(40,40))
        
        if opt["skeleton"]:
            fig, ax = plt.subplots()
            ax.axis("off")
            ax.imshow(frame)
            plotSkeleton(joint, confidences[f], {}, {}, ax, True)
            fig.canvas.draw()
            data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
            data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            writer.writeFrame(data)
            plt.close()
            plt.clf()
        else:
            writer.writeFrame(frame)
    writer.close()
def save_joint_values(joints, confidences, savename):
    file = open(savename, "wb")
    for f, joint in enumerate(joints):
        #pdb.set_trace()
        file.write(",".join(["(%f,%f,%f)" % (x,y,c) for (x,y,c) in zip(joint[0], joint[1], confidences[f])]) + "\n")
    return

def load_joint_file(joint_file):
    file = open(joint_file, "rb")
    joints_x = []
    joints_y = []
    confidences = []
    for l, line in enumerate(file):
        
        line_tmp = [tup.split('(')[1].split(',') for tup in line.split(")")[:-1]]
	joints_x.append(np.array([float(tup[0]) for tup in line_tmp]))
	joints_y.append(np.array([float(tup[1]) for tup in line_tmp]))
        confidences.append([float(tup[2]) for tup in line_tmp])
    return np.transpose(np.array([joints_x,joints_y]), (1,0,2)), np.array(confidences)
