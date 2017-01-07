__author__ = 'wangnxr'
from heatmapToJoints import heatmapToJoints
import numpy as np
import skvideo
import subprocess
import os
import glob
import argparse
import pdb
import shutil
import cv2
import scipy
import matplotlib.pyplot as plt
import time
import copy
from skimage.feature import peak_local_max

# def calc_flow_video(vid, heatmaps, opt):
#     joints_list = []
#     for f, frame in enumerate(vid):
#         warped_heatmap = {}
#         for f_p, frame_prev in enumerate(vid[f-5:f]):
#             warped_heatmap[f-f_p] = deepflow.warp_image(heatmaps[f-f_p], *deepflow.calc_flow(frame_prev, frame))
#         for f_f, frame_future in enumerate(vid[f:f+5]):
#             warped_heatmap[f-f_f] = deepflow.warp_image(heatmaps[f+f_f], *deepflow.calc_flow(frame, frame_future))
#         mean_heatmap = warp_heatmap_mean(warped_heatmap, f)
#         joints = heatmapToJoints(mean_heatmap, opt["numJoints"])
#         joints_list.append(joints)
#
#     return joints_list
#
#
#
# def warp_heatmap_mean(vid_name, heatmaps, flo_fldr):
#     mean_heatmap = np.zeros(shape=heatmaps[0].shape)
#     for key, value in heatmaps.iteritems():
#         mean_heatmap += (1/float(np.abs(f-key))) * value
#     total_weight = [(1/float(np.abs(f-key))) for key in heatmaps.iterkeys()].sum()
#     mean_heatmap /= total_weight
#     return mean_heatmap

# WARNING: this will work on little-endian architectures (eg Intel x86) only!
def load_flo_file(file):

    magic = np.fromfile(file, np.float32, count=1)
    if 202021.25 != magic:
        print 'Magic number incorrect. Invalid .flo file'
        return
    else:
        w = np.fromfile(file, np.int32, count=2)[-1]
        h = np.fromfile(file, np.int32, count=3)[-1]
        data = np.fromfile(file, np.float32, count=2*w*h+3)[3:]
        # Reshape data into 3D array (columns, rows, bands)
        data2D = np.resize(data, (h,w, 2))


    return data2D

def warp_image(im, flo_field, opt):
    resized_y, resized_x = flo_field
    resized_x = np.transpose(resized_x)
    resized_y = np.transpose(resized_y)
    if opt["warp_all"]:
        [x,y] = np.meshgrid(np.arange(im.shape[0]),np.arange(im.shape[1]))
        xx = np.ndarray.flatten(x-resized_x)
        yy = np.ndarray.flatten(y-resized_y)
        #pdb.set_trace()
        warp_func = scipy.interpolate.RectBivariateSpline(np.arange(im.shape[0]), np.arange(im.shape[1]), im)
        warped = warp_func.ev(xx, yy)
        warped = np.reshape(warped, (opt["dims"][0],opt["dims"][1]))
    else:
        warped = copy.copy(np.transpose(im))
        candidates = peak_local_max(im, min_distance = 20, threshold_abs=0.01)
        max_warp_x = max(2,2*int(np.ceil(np.fabs(resized_x.max()))))
        max_warp_y = max(2,2*int(np.ceil(np.fabs(resized_y.max()))))
        for (c_y,c_x) in candidates[::-1]:
            c_x0 = max(0,c_y-max_warp_x)
            c_x1 = min(im.shape[0],c_y+max_warp_x)
            c_y0 = max(0,c_x-max_warp_y)
            c_y1 = min(im.shape[1],c_x+max_warp_y)
            w = c_x1-c_x0
            h = c_y1-c_y0
	    #pdb.set_trace()
            [x,y] = np.meshgrid(np.arange(w),np.arange(h))
            xx = np.ndarray.flatten(x-resized_x[c_y0:c_y1, c_x0:c_x1])
            yy = np.ndarray.flatten(y-resized_y[c_y0:c_y1, c_x0:c_x1])
            #if max(abs(np.ndarray.flatten(resized_y[c_x0:c_x1, c_y0:c_y1])))>2:  		pdb.set_trace()
            warp_func = scipy.interpolate.RectBivariateSpline(np.arange(w), np.arange(h), im[c_x0:c_x1, c_y0:c_y1])
            warped_temp = warp_func.ev(xx,yy)
            warped[c_y0:c_y1, c_x0:c_x1] = np.reshape(warped_temp, (h,w))
        #print np.where(warped==warped.max())
        #print np.where(im==im.max())
        #pdb.set_trace()          
    return warped

def global_adjust(heatmaps):
    candidates = np.zeros(shape=(heatmaps.shape[0], heatmaps.shape[-1], 3, 2), dtype=int)-100
    for h, heatmap in enumerate(heatmaps):
        for l in xrange(heatmap.shape[2]):
            peaks = peak_local_max(heatmap[:,:,l], min_distance = 10, threshold_abs=0.01, num_peaks=3)
            candidates[h,l,:len(peaks)] = peaks
    joints = np.zeros(shape=(heatmaps.shape[0], 2, heatmaps.shape[-1])) 
    confidences = np.zeros(shape=(heatmaps.shape[0], heatmaps.shape[-1]))
    for h, heatmap in enumerate(heatmaps):
        #if h >441:
        #    pdb.set_trace()
        hmin = max(0,h-20)
        hmax = min(len(heatmaps),h+20)
        for l in xrange(heatmap.shape[2]):
            try:
            	nearby_vals =  [np.median(np.max(np.max(heatmaps[hmin:hmax,(candidates[h,l,c,0]-20):(candidates[h,l,c,0]+20),(candidates[h,l,c,1]-5):(candidates[h,l,c,1]+5),l], axis=1), axis=1)) for c in xrange(3) if candidates[h,l,c,0]>0]
	    except:
		nearby_vals = []                

    #nearby_candidates = (abs(candidates[hmin:hmax,l,:,0]-candidate[0])<20) & (abs(candidates[hmin:hmax,l,:,1]-candidate[1])<20)
                    #filter_candidates = (tuple(np.where(nearby_candidates)[0]),tuple(candidates[hmin:hmax,l][nearby_candidates][:,0]),tuple(candidates[hmin:hmax,l][nearby_candidates][:,1]))
                    #if h==30:
		    #	pdb.set_trace()
            if len(nearby_vals)>0:
	    	joints[h,::-1,l]=candidates[h,l,np.array(nearby_vals).argmax(),:]
            else:
                joints[h,::-1,l]=np.unravel_index(heatmap[:,:,l].argmax(), heatmap[:,:,l].shape)
	    #pdb.set_trace()
	    confidences[h,l]=heatmap[int(joints[h,1,l]),int(joints[h,0,l]),l]
            #pdb.set_trace()
    return joints, confidences
def calc_flow_video(vid_name, heatmaps, opt):
	
    joints_list = []
    confidences_list = []
    tmp_res_fldr = "tmp_%s" % vid_name
    if opt["skeleton"]:
        joints_to_do = opt["numJoints"]
    else:

        joints_to_do = 1
    if not os.path.exists(tmp_res_fldr):
        os.makedirs(tmp_res_fldr)
    if not os.path.exists("%s/%s/" % (opt["floDir"], vid_name)):
        if os.path.exists("%s/%s.avi"):
            get_video_flo("%s/%s.avi" % (opt["inputDir"], vid_name), opt["floDir"])
        else:
            get_video_flo("%s/%s.mp4" % (opt["inputDir"], vid_name), opt["floDir"])
    mean_heatmaps = []
    for f, frame in enumerate(heatmaps):
        cnt = 1
        if f%100==0:
            print "warping frame %i" % f
        warped_imgs = [1*frame[:,:,:joints_to_do]]
        for f2 in range(max(0, f-1), min(len(heatmaps), f+2)):
            if not f==f2:
                flo_file = "%s/%s/%i_%i.flo" % (opt["floDir"], vid_name, f2, f)
                flo_data = load_flo_file(flo_file)

                flo_data = [cv2.resize(flo_data[:,:,0], (opt["dims"][0],opt["dims"][1])),cv2.resize(flo_data[:,:,1], (opt["dims"][0],opt["dims"][1]))]
                if f2<f:
                    warped_imgs.append(np.transpose(np.array([5*warp_image(mean_heatmaps[f2][:,:,j], flo_data, opt) for j in xrange(joints_to_do)])))
                    cnt +=5
                else:
                    warped_imgs.append(np.transpose(np.array([warp_image(heatmaps[f2][:,:,j], flo_data, opt) for j in xrange(joints_to_do)])))
                    cnt +=1
                    #if (flo_data[0].max() >2) and (heatmaps[f2][:,:,0].max() > 0.1):
                    #if f==168:
		    #    plt.imsave("heatmapf2.png", heatmaps[f2][:,:,6])
                    #    plt.imsave("warp1.png", warped_imgs[0][:,:,6])
		    #	plt.imsave("warp2.png", warped_imgs[1][:,:,6])
                    #    plt.imsave("warp3.png", warped_imgs[2][:,:,6])

                     #   pdb.set_trace()
                      #  warp_image(heatmaps[f2][:,:,6], flo_data, opt)
        #if f > 1:
        #    plt.imshow(np.concatenate([warped_imgs[0], warped_imgs[1]/10, warped_imgs[2]])[:,:,1])
        #    print warped_imgs[0][:,:,4].max()
        #    plt.show()
        warped_imgs = np.array(warped_imgs)
        mean_heatmaps.append(np.sum(warped_imgs, axis=0)/cnt)
        
    #pdb.set_trace()
    joints_list, confidences_list = global_adjust(np.array(mean_heatmaps))
    #for f, frame in enumerate(mean_heatmaps):
    #	joints, confidences = heatmapToJoints(frame, joints_to_do)
    #    joints_list.append(joints)
    #    confidences_list.append(confidences)

    shutil.rmtree("%s/%s/" % (opt["floDir"], vid_name))    
    return joints_list, confidences_list

def get_video_flo(vid_fname, save_loc):

    vid_name = vid_fname.split("/")[-1].split(".")[0]
    if not os.path.exists(save_loc + "/" + vid_name):
        if not os.path.exists("tmp_%s" % vid_name):
            os.makedirs("tmp_%s" % vid_name)
        if not os.path.exists(save_loc + "/" + vid_name):
            os.makedirs(save_loc + "/" + vid_name)
        subprocess.call("ffmpeg -i " + vid_fname + " -f image2 -r 30 tmp_" + vid_name + "/%04d.png  ", shell=True)
        filelist = sorted(glob.glob(os.getcwd() + "/tmp_%s/*.png" % vid_name))
        input = {"img1":[], "img2":[], "save":[]}
        for f, frame_name in enumerate(filelist):
            for f2 in range(max(0,f-1),min(f+2, len(filelist))):
                if not f2 == f:
                    f2_name = "/".join(frame_name.split("/")[:-1]) + "/%04d.png" % (f2+1)
                    save_name = "%s/%s/%i_%i.flo" % (save_loc, vid_name, f, f2)
                    input["img1"].append(frame_name)
                    input["img2"].append(f2_name)
                    input["save"].append(save_name)
        with open("tmp_%s/img1.txt" % vid_name, "wb")as img1:
            img1.write("\n".join(input["img1"]))
        with open("tmp_%s/img2.txt" % vid_name, "wb")as img2:
            img2.write("\n".join(input["img2"]))
        subprocess.call("rm /home/wangnxr/Documents/flownet-release/models/flownet/*.flo", shell=True)
        subprocess.call("python /home/wangnxr/Documents/flownet-release/models/flownet/demo_flownet.py "
                            "C %s/tmp_%s/img1.txt %s/tmp_%s/img2.txt %s/tmp_%s/" 
			     % (os.getcwd(),vid_name,os.getcwd(), vid_name, os.getcwd(), vid_name), shell=True)
        for f, file in enumerate(sorted(glob.glob("/home/wangnxr/Documents/flownet-release/models/flownet/*.flo"))):
            shutil.move(file, input["save"][f])
        shutil.rmtree("tmp_%s" % vid_name)

if __name__=="__main__":
   
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', required=True, help="Video Directory")
    parser.add_argument('-s', '--save', required=True, help="Save directory")
    args = parser.parse_args()
    for file in sorted(glob.glob(args.file+"/*")):
    	get_video_flo(file, args.save)
