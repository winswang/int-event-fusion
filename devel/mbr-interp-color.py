import tensorflow as tf
import matplotlib.pyplot as plt
import os, sys, inspect
import numpy as np
import datetime, time
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_psnr as psnr
from skimage.io import imread
from skimage.io import imsave
import utils.visualization as vs
import utils.events_processing as ep
from utils.tensorflowvgg import vgg19
import json

# load data (color frames and events)
dataDir = '/home/winston/git/int-event-fusion/sample_preparation/adobe-test'
vidList = os.listdir(dataDir)
vidId = 1
clipDir = os.path.join(dataDir, vidList[vidId], 'clip')
evfDir = os.path.join(dataDir, vidList[vidId], 'event-frames')
startFrameId = 0
endFrameId = 24
vid_gt = np.array([imread(os.path.join(clipDir, '%02d.png' % i)) for i in range(startFrameId, endFrameId+1)])/255.0
print("Finished reading video 'vid_gt', with shape", np.shape(vid_gt), "max_val:", np.max(vid_gt))
evf_gt = (np.array([imread(os.path.join(evfDir, '%02d.png' % i)) for i in range(startFrameId, endFrameId)])/255.0 - 0.5)*2
print("Finished reading event frames 'evf_gt', with shape", np.shape(evf_gt), "max_val", np.max(evf_gt), "min_val", np.min(evf_gt))
lres_gt = np.array([vid_gt[0], vid_gt[-1]])
print("Finished creating 'lres_gt', with shape", np.shape(lres_gt))

# models
def init_hres(config, lres_gt):
    # input lres_gt should be in shape (t, y, x, c)
    lres_mean = np.expand_dims(np.mean(lres_gt, axis = 0), axis = 0)
    #print("shape of 'lres_mean'", np.shape(lres_mean))
    # (1, y, x, c)
    lres_start = tf.Variable(np.expand_dims(lres_gt[0], axis = 0), dtype = tf.float32)
    lres_end = tf.Variable(np.expand_dims(lres_gt[1], axis = 0), dtype = tf.float32)
    # init hres
    if config.init_mode == 0: # hres init as lres_0-lres_mean-lres_1
        for i in range(config.dim_hrt):
            if i == 0:
                hres_init = lres_start
            elif i == config.dim_hrt - 1:
                hres_init = tf.concat([hres_init, tf.Variable(lres_end, dtype = tf.float32)], 0)
            else:
                hres_init = tf.concat([hres_init, tf.Variable(lres_mean_var, dtype = tf.float32)], 0)
        print("hres_tensor initialized (lres-mean-lres)")
    elif config.init_mode == 1: # hres init as mean+rand
        for i in range(config.dim_hrt):
            lres_rand = lres_mean*(1.0 - config.noise_a) + config.noise_a*(np.random.rand(1, config.dim_y, config.dim_x, config.dim_c)-0.5)
            if i == 0:
                hres_init = lres_start
            elif i == config.dim_hrt - 1:
                hres_init = tf.concat([hres_init, lres_end], 0)
            else:
                hres_init = tf.concat([hres_init, tf.Variable(lres_rand, dtype = tf.float32)], 0)
        print("hres_tensor initialized (rand)")
    elif config.init_mode == 2: # a* lres_start + (1-a)*lres_end
        for i in range(config.dim_hrt):
            a = i / (config.dim_hrt - 1)
            if i == 0:
                hres_init = lres_start
            else:
                frame = (1-a)*lres_start + a*lres_end
                hres_init = tf.concat([hres_init, tf.Variable(frame, dtype = tf.float32)], 0)
    hres_init = tf.expand_dims(hres_init, axis = 0)
    print("hres_init shape:", np.shape(hres_init))
    return hres_init
    
def init_flow(config):
    if config.flow_init == 0:
        flow_x = tf.Variable(tf.zeros(config.hres_dim))
        flow_y = tf.Variable(tf.zeros(config.hres_dim))
        flow = tf.concat([flow_x, flow_y], 0)
        return flow
        
def frame_model(config, hres_tensor):
    if config.shape_mode <=1: #  interpolation
        # hres_tensor in shape (t, y, x, c)
        lres_tensor = tf.transpose(tf.gather_nd(tf.transpose(hres_tensor, perm = [1,2,3,4,0]), 
                                                indices = [[0],[config.dim_hrt-1]]), 
                                   perm = [4,0,1,2,3])
        print("lres_tensor, shape:", np.shape(lres_tensor))
    return lres_tensor

def event_model(config, hres_tensor):
    hres_relu = tf.nn.relu(hres_tensor)
    tanh_coef = tf.constant(config.tanh_coef)
    kernel = tf.constant([-1,1], dtype = tf.float32)
    k_tyx = tf.expand_dims(tf.expand_dims(tf.expand_dims(kernel, axis = 1), axis = 2), axis = 3)
    k_tyxi = tf.expand_dims(tf.concat([k_tyx, k_tyx, k_tyx], axis = 3), axis = 4)
    k_tyxio = tf.concat([k_tyxi, k_tyxi, k_tyxi], axis = 4)
    #print("shape of 'hres_ndhwc':", np.shape(hres_ndhwc))
    #print("shape of 'k_tyxio':", np.shape(k_tyxio))
    evf_tanh = tf.tanh(tanh_coef*tf.nn.convolution(input = hres_relu, filter = k_tyxio, padding = "VALID", data_format = "NDHWC"))
    print("evf_tanh, shape:", np.shape(evf_tanh))
    if config.shape_mode == 0:
        evf_tanh = tf.reduce_sum(evf_tanh, 1, keepdims = True)
    return evf_tanh

def tv_2d(config, hres_tensor):
    kernel = tf.constant([-1,1], dtype = tf.float32)
    kx = tf.expand_dims(tf.expand_dims(tf.expand_dims(kernel, axis = 0), axis = 2), axis = 3)
    kxi = tf.expand_dims(tf.concat([kx, kx, kx], axis = 3), axis = 4)
    kxio = tf.concat([kxi, kxi, kxi], axis = 4)
    ky = tf.expand_dims(tf.expand_dims(tf.expand_dims(kernel, axis = 1), axis = 2), axis = 3)
    kyi = tf.expand_dims(tf.concat([ky, ky, ky], axis = 3), axis = 4)
    kyio = tf.concat([kyi, kyi, kyi], axis = 4)
    
    dx = tf.nn.convolution(input = hres_tensor, filter = kxio, padding = "SAME", data_format = "NDHWC")
    dy = tf.nn.convolution(input = hres_tensor, filter = kyio, padding = "SAME", data_format = "NDHWC")
    return tf.norm(dx+dy, ord = 1)

def tv_t(config, hres_tensor):
    kernel = tf.constant([-1,1], dtype = tf.float32)
    kt = tf.expand_dims(tf.expand_dims(tf.expand_dims(kernel, axis = 0), axis = 0), axis = 3)
    kti = tf.expand_dims(tf.concat([kt, kt, kt], axis = 3), axis = 4)
    ktio = tf.concat([kti, kti, kti], axis = 4)
    dt = tf.nn.convolution(input = hres_tensor, filter = ktio, padding = "SAME", data_format = "NDHWC")
    return tf.norm(dt, ord = 1)


def loss_all(config, ph, hres_tensor, lres_tensor, evf_tensor, flow_tensor):
    # frame_loss = tf.reduce_mean(tf.squared_difference(ph.lres_gt, lres_tensor))
    # event_loss = tf.constant(config.ev_weight)*tf.reduce_mean(tf.squared_difference(ph.evf_gt, evf_tensor))
    frame_loss = tf.norm(ph.lres_gt - lres_tensor, ord = 1)
    event_loss = tf.constant(config.ev_weight)*tf.norm(ph.evf_gt - evf_tensor, ord = 1)
    tv_loss = tf.constant(config.tv_coef_xy)*tv_2d(config, hres_tensor) + tf.constant(config.tv_coef_t)*tv_t(config, hres_tensor)
    opt_flow_loss = tf.constant(config.flow_loss_coef)*flow_loss(config, hres_tensor, flow_tensor)
    return frame_loss + event_loss + tv_loss + opt_flow_loss

def loss_pix_tv(config, ph, hres_tensor, lres_tensor, evf_tensor):
    frame_loss = tf.norm(ph.lres_gt - tf.squeeze(lres_tensor, [0]), ord = 1)
    event_loss = tf.constant(config.ev_weight)*tf.norm(ph.evf_gt - tf.squeeze(evf_tensor, [0]), ord = 1)
    tv_loss = tf.constant(config.tv_coef_xy)*tv_2d(config, hres_tensor) + tf.constant(config.tv_coef_t)*tv_t(config, hres_tensor)
    return frame_loss + event_loss + tv_loss

# config
class config():
    def __init__(self, hres_dim = None, evf_dim = None, shape_mode = 0, init_mode = 0):
        self.shape_mode = shape_mode
        self.init_mode = init_mode
                
        if self.shape_mode == 0: # interpolation mode, lres is {start + end} frame of hres, evf is 1-frame
            self.hres_dim = hres_dim
            (self.dim_hrt, self.dim_y, self.dim_x, self.dim_c) = hres_dim
            self.lres_dim = (2, self.dim_y, self.dim_x, self.dim_c)
            self.evf_dim = (self.dim_hrt - 1, self.dim_y, self.dim_x, self.dim_c)
        elif self.shape_mode == 1: # interpolation mode, lres same as mode 0, evf will be defined
            self.evf_dim = evf_dim
            (self.dim_evt, self.dim_y, self.dim_x, self.dim_c) = evf_dim
            self.lres_dim = (2, self.dim_y, self.dim_x, self.dim_c)
            
            self.dim_hrt = self.dim_evt + 1
            
        if self.init_mode == 1: # random initialization of hres_tensor
            self.noise_a = 1e-4 # noise magnitude
        
        # events
        self.ev_weight = 9e-2
        self.tanh_coef = 5.0
        
        # hres_tv
        self.tv_coef_xy = 6e-3
        self.tv_coef_t = 0e-3

        # learning
        self.lr_init = 8e-3
        self.lr_update = 100
        self.epochs = 250
        self.beta1 = 0.9
        self.beta2 = 0.99


# place holder
class place_holder():
    def __init__(self, config):
        self.learning_rate = tf.placeholder(tf.float32)
        self.lres_gt = tf.placeholder(tf.float32, shape = config.lres_dim)
        self.evf_gt = tf.placeholder(tf.float32, shape = config.evf_dim)
        
evf_dim = np.shape(evf_gt)
res_config = config(evf_dim = evf_dim, shape_mode = 1, init_mode = 2)
res_ph = place_holder(res_config)

hres_tensor = init_hres(res_config, lres_gt)
lres_tensor = frame_model(res_config, hres_tensor)
evf_tensor = event_model(res_config, hres_tensor)
loss = loss_pix_tv(res_config, res_ph, hres_tensor, lres_tensor, evf_tensor)

optimizer = tf.train.AdamOptimizer(learning_rate = res_ph.learning_rate, beta1 = res_config.beta1, beta2 = res_config.beta2)
opt_min = optimizer.minimize(loss)

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
sess.run(tf.global_variables_initializer())

iloss = np.empty((res_config.epochs,))
for iepoch in range(res_config.epochs):
    if iepoch == 0:
        res_config.lr = res_config.lr_init
    hres_rec, iloss[iepoch], _ = sess.run([hres_tensor, loss, opt_min],
                               feed_dict={res_ph.lres_gt: lres_gt, 
                                          res_ph.evf_gt: evf_gt, 
                                          res_ph.learning_rate: res_config.lr})
    if iepoch % res_config.lr_update == 0:
        print("Epoch:", iepoch, "learning rate: %5f" % res_config.lr)
        if iepoch > 0:
            res_config.lr *= 0.1
print("hres_rec shape:", np.shape(hres_rec))
print("Max value %0.2f, min value %0.2f" % (np.amax(hres_rec), np.amin(hres_rec)))

hres_psnr, hres_ssim = vs.computePsnrSsim(vid_gt, hres_rec)

savePath = 'results/mbr-interp-psnr-ssim'
if not os.path.isdir(savePath):
    os.mkdir(savePath)

with open(os.path.join(savePath, vidList[vidId] + '.txt'), 'a') as f:
    saveConfig = {'shape_mode': res_config.shape_mode, 'init_mode': res_config.init_mode,
                 'ev_weight': res_config.ev_weight,
                  'tanh_coef': res_config.tanh_coef, 'tv_coef_xy': res_config.tv_coef_xy,
                 'tv_coef_t': res_config.tv_coef_t, 'lr': res_config.lr_init, 'lr_update': res_config.lr_update,
                  'epochs': res_config.epochs, 'psnr': hres_psnr, 'ssim': hres_ssim}
    json.dump(saveConfig, f)
    f.write('\n\n')
