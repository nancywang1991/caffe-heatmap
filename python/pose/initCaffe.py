import caffe
import pdb

# Initialise Caffe
def initCaffe(opt):
    caffe.set_mode_gpu()
    gpu_id = opt["gpu_id"]
    caffe.set_device(gpu_id)

    net = caffe.Net(opt["modelDefFile"], opt["modelFile"], caffe.TEST)
    return net
