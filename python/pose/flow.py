__author__ = 'wangnxr'
import deepflow.py_fastdeepflow.fastdeepflow as deepflow
from heatmapToJoints import heatmapToJoints
import numpy as np

def calc_flow_video(vid, heatmaps, opt):
    joints_list = []
    for f, frame in enumerate(vid):
        warped_heatmap = {}
        for f_p, frame_prev in enumerate(vid[f-5:f]):
            warped_heatmap[f-f_p] = deepflow.warp_image(heatmaps[f-f_p], *deepflow.calc_flow(frame_prev, frame))
        for f_f, frame_future in enumerate(vid[f:f+5]):
            warped_heatmap[f-f_f] = deepflow.warp_image(heatmaps[f+f_f], *deepflow.calc_flow(frame, frame_future))
        mean_heatmap = warp_heatmap_mean(warped_heatmap, f)
        joints = heatmapToJoints(mean_heatmap, opt["numJoints"])
        joints_list.append(joints)

    return joints_list



def warp_heatmap_mean(heatmaps, f):
    mean_heatmap = np.zeros(shape=heatmaps[0].shape)
    for key, value in heatmaps.iteritems():
        mean_heatmap += (1/float(np.abs(f-key))) * value
    total_weight = [(1/float(np.abs(f-key))) for key in heatmaps.iterkeys()].sum()
    mean_heatmap /= total_weight
    return mean_heatmap