# Plotskeleton
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import pdb


def plotSkeleton(j, opts, handle, dominantOnly=False):

    if len(opts) == 0:
        opts = plotSkeletonDefaultopts(opts)
    if 'jointlinewidth' not in opts.keys():
        opts["jointlinewidth"] = 1
    if 'jointlinecolor' not in opts.keys():
        opts["jointlinecolor"] = np.zeros(shape=(7, 3))
    if not hasattr(opts["jointsize"], "__len__"):
        opts["jointsize"] = opts["jointsize"] * np.zeros(shape=(7, 1)) + 1

    if not hasattr(opts["jointlinewidth"], "__len__"):
        opts["jointlinewidth"] = opts["jointlinewidth"] * np.zeros(shape=(7, 1)) + 1

    joints = range(7)

    # wrist only plot
    if len(handle) == 0:
        if j.shape[1] == 2:
            joints = range(2)
            dontPlotSkeleton = True
        else:
            dontPlotSkeleton = False

        if j.shape[1] == 3:
            joints = range(3)
            dontPlotSkeleton = True

        if ~dontPlotSkeleton:
            # draw skelton
            if ~dominantOnly:
                handle["ula"] = plt.plot(j[0, [4, 6]], j[1, [4, 6]], linewidth=opts["linewidth"], color=opts["clr"][8])
            handle["ura"] = plt.plot(j[0, [3, 5]], j[1, [3, 5]],  linewidth=opts["linewidth"], color=opts["clr"][8])
            if ~dominantOnly:
                handle["lla"] = plt.plot(j[0, [2, 4]], j[1, [2, 4]], linewidth=opts["linewidth"], color=opts["clr"][9])
            handle["lra"] = plt.plot(j[0, [1, 3]], j[1, [1, 3]], linewidth=opts["linewidth"], color=opts["clr"][9])

        # draw joints
        if dominantOnly:
            joints=[0, 1, 3, 5]
        handle["joints"] = []
        for c in joints:
            handle["joints"].append(plt.plot(j[0, c], j[1, c], markerfacecolor=opts["clr"][c], markersize = opts["jointsize"][c],
                                        linewidth=opts["jointlinewidth"][c], color=opts["jointlinecolor"][c]))

    else:
        # draw skelton
        set(handle["lla"], 'xdata', j[0, [2, 4]], 'ydata', j[1, [2, 4]]);
        set(handle["ula"], 'xdata', j[0, [4, 6]], 'ydata', j[1, [4, 6]]);
        set(handle["lra"], 'xdata', j[0, [1, 3]], 'ydata', j[1, [1, 3]]);
        set(handle["ura"], 'xdata', j[0, [3, 5]], 'ydata', j[1, [3, 5]]);
        # draw joints
        for c in xrange(7):
            set(handle["joints"][c], 'xdata', j[0, c], 'ydata', j[1, c]);
    return handle

def plotSkeletonDefaultopts(opts):

    opts["clr"] = [cmx.jet(x)[1:] for x in xrange(10)]
    #sets coulour of joints
    opts["clr"][8] = (1,0,0)
    opts["clr"][9] = (0,1,0)
    opts["linewidth"] = 2
    opts["jointsize"] = 6
    return opts
