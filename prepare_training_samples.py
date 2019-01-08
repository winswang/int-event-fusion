import numpy as np
import tensorflow as tf
import os, sys
import random, json
from keras.preprocessing.image import img_to_array, load_img
import matplotlib.pyplot as plt
from keras.layers import  Input,Conv2D,BatchNormalization,Activation,Subtract
from keras.models import Model, load_model
from keras.optimizers import Adam
import keras.backend as K
import time

# models
def init_hres(config, lres_gt):
    lres_gt_var = tf.Variable(lres_gt)
    # init hres
    if config.hres_init_type == 0:
        hres_partial = tf.Variable(tf.random_uniform(config.evf_dim))
        if config.lres_type == 0:
            print("hres initialized, frame 0 == reference")
            return tf.concat([lres_gt_var, hres_partial], 3)
        elif config.lres_type == 1:
            print("hres initialized, frame 1 == reference")
            return tf.concat([hres_partial, lres_gt_var], 3)
    elif config.hres_init_type == 1:
        for i in range(config.ev_dim):
            lres_gt_var = tf.concat([lres_gt_var, tf.Variable(lres_gt)], 3)
        return lres_gt_var
    
def init_flow(config):
    if config.flow_init == 0:
        flow_x = tf.Variable(tf.zeros(config.hres_dim))
        flow_y = tf.Variable(tf.zeros(config.hres_dim))
        flow = tf.concat([flow_x, flow_y], 0)
        return flow
        
def frame_model(config, hres_tensor):
    hres_transpose = tf.transpose(hres_tensor, perm = [3,0,1,2])
    if config.lres_type == 0:
        lres_transpose = tf.gather_nd(hres_transpose, indices = [[0]])
    elif config.lres_type == 1:
        lres_transpose = tf.gather_nd(hres_transpose, indices = [[0]])
    return tf.transpose(lres_transpose, perm = [1,2,3,0])

def event_model(config, hres_tensor):
    tanh_coef = tf.constant(config.tanh_coef)
    kernel = tf.expand_dims(tf.expand_dims(tf.expand_dims(tf.constant([-1,1], dtype = tf.float32), dim = 1), dim = 0), dim = 0)
    return tf.tanh(tanh_coef*tf.nn.convolution(input = hres_tensor, filter = kernel, padding = "VALID", data_format = "NHWC"))

def tv_2d(config, hres_tensor):
    hres_ndhwc = tf.expand_dims(hres_tensor, dim = 4)
    kernel = tf.constant([-1,1], dtype = tf.float32)
    kx = tf.expand_dims(tf.expand_dims(tf.expand_dims(tf.expand_dims(tf.constant([-1,1], dtype = tf.float32), dim = 1), dim = 2), dim = 3), dim = 4)
    ky = tf.expand_dims(tf.expand_dims(tf.expand_dims(tf.expand_dims(tf.constant([-1,1], dtype = tf.float32), dim = 0), dim = 2), dim = 3), dim = 4)

    dx = tf.squeeze(tf.nn.convolution(input = hres_ndhwc, filter = kx, padding = "SAME", data_format = "NDHWC"), [4])
    dy = tf.squeeze(tf.nn.convolution(input = hres_ndhwc, filter = ky, padding = "SAME", data_format = "NDHWC"), [4])
    return tf.norm(dx+dy, ord = 1)

def tv_t(config, hres_tensor):
    hres_ndhwc = tf.expand_dims(hres_tensor, dim = 4)
    kernel = tf.constant([-1,1], dtype = tf.float32)
    kt = tf.expand_dims(tf.expand_dims(tf.expand_dims(tf.expand_dims(tf.constant([-1,1], dtype = tf.float32), dim = 0), dim = 0), dim = 3), dim = 4)
    dt = tf.squeeze(tf.nn.convolution(input = hres_ndhwc, filter = kt, padding = "SAME", data_format = "NDHWC"), [4])
    return tf.norm(dt, ord = 1)

def flow_loss(config, hres_tensor, flow):
    flowx_tensor = tf.slice(flow, [0,0,0,0],[1,-1,-1,-1])
    flowy_tensor = tf.slice(flow, [1,0,0,0],[1,-1,-1,-1])
    hres_ndhwc = tf.expand_dims(hres_tensor, dim = 4)
    flowx_ndhwc = tf.expand_dims(flowx_tensor, dim = 4)
    flowy_ndhwc = tf.expand_dims(flowy_tensor, dim = 4)
    
    flow_coef_xy = tf.constant(config.flow_norm_xy_coef, dtype = tf.float32)
    flow_coef_t = tf.constant(config.flow_norm_t_coef, dtype = tf.float32)
    kernel = tf.constant([-1,1], dtype = tf.float32)
    
    kx = tf.expand_dims(tf.expand_dims(tf.expand_dims(tf.expand_dims(kernel, dim = 1), dim = 2), dim = 3), dim = 4)
    ky = tf.expand_dims(tf.expand_dims(tf.expand_dims(tf.expand_dims(kernel, dim = 0), dim = 2), dim = 3), dim = 4)
    kt = tf.expand_dims(tf.expand_dims(tf.expand_dims(tf.expand_dims(kernel, dim = 0), dim = 0), dim = 3), dim = 4)
    
    dHx = tf.squeeze(tf.nn.convolution(input = hres_ndhwc, filter = kx, padding = "SAME", data_format = "NDHWC"), [4])
    dHy = tf.squeeze(tf.nn.convolution(input = hres_ndhwc, filter = ky, padding = "SAME", data_format = "NDHWC"), [4])
    dHt = tf.squeeze(tf.nn.convolution(input = hres_ndhwc, filter = kt, padding = "SAME", data_format = "NDHWC"), [4])
    
    flow_eq = tf.multiply(dHx, flowx_tensor) + tf.multiply(dHy, flowy_tensor) + dHt
    
    dUxx = tf.squeeze(tf.nn.convolution(input = flowx_ndhwc, filter = kx, padding = "SAME", data_format = "NDHWC"), [4])
    dUxy = tf.squeeze(tf.nn.convolution(input = flowx_ndhwc, filter = ky, padding = "SAME", data_format = "NDHWC"), [4])
    dUyx = tf.squeeze(tf.nn.convolution(input = flowy_ndhwc, filter = kx, padding = "SAME", data_format = "NDHWC"), [4])
    dUyy = tf.squeeze(tf.nn.convolution(input = flowy_ndhwc, filter = ky, padding = "SAME", data_format = "NDHWC"), [4])
    dUxt = tf.squeeze(tf.nn.convolution(input = flowx_ndhwc, filter = kt, padding = "SAME", data_format = "NDHWC"), [4])
    dUyt = tf.squeeze(tf.nn.convolution(input = flowy_ndhwc, filter = kt, padding = "SAME", data_format = "NDHWC"), [4])
    
    return tf.norm(flow_eq, ord = 1) + flow_coef_xy*tf.norm(dUxx+dUxy+dUyx+dUyy, ord = 1)+ flow_coef_t*tf.norm(dUxt+dUyt, ord = 1)


def loss_all(config, ph, hres_tensor, lres_tensor, evf_tensor, flow_tensor):
    # frame_loss = tf.reduce_mean(tf.squared_difference(ph.lres_gt, lres_tensor))
    # event_loss = tf.constant(config.ev_weight)*tf.reduce_mean(tf.squared_difference(ph.evf_gt, evf_tensor))
    frame_loss = tf.norm(ph.lres_gt - lres_tensor, ord = 1)
    event_loss = tf.constant(config.ev_weight)*tf.norm(ph.evf_gt - evf_tensor, ord = 1)
    tv_loss = tf.constant(config.tv_coef_xy)*tv_2d(config, hres_tensor) + tf.constant(config.tv_coef_t)*tv_t(config, hres_tensor)
    opt_flow_loss = tf.constant(config.flow_loss_coef)*flow_loss(config, hres_tensor, flow_tensor)
    return frame_loss + event_loss + tv_loss + opt_flow_loss

def loss_no_flow(config, ph, hres_tensor, lres_tensor, evf_tensor):
    # frame_loss = tf.reduce_mean(tf.squared_difference(ph.lres_gt, lres_tensor))
    # event_loss = tf.constant(config.ev_weight)*tf.reduce_mean(tf.squared_difference(ph.evf_gt, evf_tensor))
    frame_loss = tf.norm(ph.lres_gt - lres_tensor, ord = 1)
    event_loss = tf.constant(config.ev_weight)*tf.norm(ph.evf_gt - evf_tensor, ord = 1)
    tv_loss = tf.constant(config.tv_coef_xy)*tv_2d(config, hres_tensor) + tf.constant(config.tv_coef_t)*tv_t(config, hres_tensor)
    return frame_loss + event_loss + tv_loss

def img_diff_pair(img1, img2, event_thres = None):
    if event_thres == None:
        event_thres = 0.8*1e-1
    frame_diff = np.log(img2+1e-10) - np.log(img1+1e-10)
    evf = np.zeros_like(frame_diff,dtype=np.float)
    evf[frame_diff>event_thres] = 1.0
    evf[frame_diff<-event_thres] = -1.0
    return evf

def split_train_test(class_list):
    t_num = 0
    v_num = 0
    for iclass in class_list:
        if random.random() <= 0.8:
            # train
            if t_num == 0:
                t_list = np.asarray([iclass])
                t_num = t_num + 1
            else:
                t_list = np.concatenate((t_list, np.asarray([iclass])), axis = 0)
        else:
            # test
            if v_num == 0:
                v_list = np.asarray([iclass])
                v_num = v_num + 1
            else:
                v_list = np.concatenate((v_list, np.asarray([iclass])), axis = 0)
    return (t_list, v_list)

def pairname_list(a_list, num = 400):
    data_dir = '/data/NfS/'
    fps = '240'
    for isamp in range(int(num)):
        class_idx = random.randint(0, len(a_list)-1)
        path = os.path.join(data_dir, a_list[class_idx], fps, a_list[class_idx])
        img_list = sorted(os.listdir(path))
        img_idx = random.randint(1,len(img_list)-2)
        img_p1 = os.path.join(path, img_list[img_idx])
        img_p2 = os.path.join(path, img_list[img_idx+1])
        temp_list = np.expand_dims(np.asarray([img_p1, img_p2]), axis = 0) 
        if isamp == 0:
            img_pair = temp_list
        else:
            img_pair = np.concatenate((img_pair,temp_list), axis = 0)
    
    return img_pair

def read_crop_pair(pairname, dim = (180, 240)):
    dim1, dim2 = dim
    img1 = img_to_array(load_img(pairname[0], color_mode = "grayscale"))/255.
    img2 = img_to_array(load_img(pairname[1], color_mode = "grayscale"))/255.
    
    trial = 0
    ev_nz = 0
    while (ev_nz > 0.2*dim1*dim2 or ev_nz == 0) and trial < 50:
        idx1 = random.randint(0, np.size(img1,0) - dim1 - 1)
        idx2 = random.randint(0, np.size(img1,1) - dim2 - 1)
        ev_th = random.random()*0.05
        tmp1 = img1[idx1:idx1+dim1,idx2:idx2+dim2,:]
        tmp2 = img2[idx1:idx1+dim1,idx2:idx2+dim2,:]
        evf = img_diff_pair(tmp1, tmp2, ev_th)
        ev_nz = np.count_nonzero(evf)
        trial = trial + 1
        
    hres_gt = np.expand_dims(np.dstack((tmp1, tmp2)), 0) 
    evf_gt = np.expand_dims(evf, 0)
    return (hres_gt, evf_gt)

def prepare_hres_evf(a_list, dim):
    data_dir = '/data/NfS/'
    fps = '240'
    dim1, dim2 = dim
    
    class_idx = random.randint(0, len(a_list)-1)
    path = os.path.join(data_dir, a_list[class_idx], fps, a_list[class_idx])
    img_list = sorted(os.listdir(path))
    img_idx = random.randint(1,len(img_list)-2)
    
    # randomly generate paths
    img_p1 = os.path.join(path, img_list[img_idx])
    if random.random() > 0.5:
        img_p2 = os.path.join(path, img_list[img_idx+1])
    else:
        img_p2 = os.path.join(path, img_list[img_idx-1])
        
    # load_images
    img1 = img_to_array(load_img(img_p1, color_mode = "grayscale"))/255.
    img2 = img_to_array(load_img(img_p2, color_mode = "grayscale"))/255.
    
    # crop images
    trial = 0
    ev_nz = 0
    while (ev_nz > 0.2*dim1*dim2 or ev_nz == 0) and trial < 50:
        idx1 = random.randint(0, np.size(img1,0) - dim1 - 1)
        idx2 = random.randint(0, np.size(img1,1) - dim2 - 1)
        ev_th = random.random()*0.05
        tmp1 = img1[idx1:idx1+dim1,idx2:idx2+dim2,:]
        tmp2 = img2[idx1:idx1+dim1,idx2:idx2+dim2,:]
        evf = img_diff_pair(tmp1, tmp2, ev_th)
        ev_nz = np.count_nonzero(evf)
        trial = trial + 1

    hres_gt = np.expand_dims(np.dstack((tmp1, tmp2)), 0) 
    lres_gt = np.expand_dims(tmp1, 0)
    evf_gt = np.expand_dims(evf, 0)
    
    return hres_gt, lres_gt, evf_gt

# config
class config():
    def __init__(self, dim = None, ev_dim = 1):
        if dim == None:
            self.dim1 = 180
            self.dim2 = 240
            self.ev_dim = 1
        else:
            self.dim1, self.dim2 = dim
            self.ev_dim = ev_dim
        self.hres_dim = (1, self.dim1, self.dim2, self.ev_dim + 1)
        self.lres_dim = (1, self.dim1, self.dim2, 1)
        self.evf_dim = (1, self.dim1, self.dim2, self.ev_dim)
        # frames
        self.hres_init_type = 1 # 0: initialize from random 1: initialize from lr video
        self.lres_type = 0  # 0: start frame; 1: end frame
        
        # events
        self.ev_weight = random.random()*4e-1+1e-1# 5e-1
        self.tanh_coef = random.random()*12.0+8.0 # 20.0
        
        # hres_tv
        self.tv_coef_xy = random.random()*5e-1+3e-1
        self.tv_coef_t = random.random()*4e-1+2e-1# 5e-1
        
        # flow
        self.flow_init = 0# Initialization
        self.flow_norm_xy_coef = 1e-1
        self.flow_norm_t_coef = 1e-1
        self.flow_loss_coef = 0.0#1e-1

        # learning
        self.lr = random.random()*2.5e-3+6.5e-3
        self.lr_decay = 0.9
        self.epochs = random.randint(1, 250)
        self.beta1 = 0.9
        self.beta2 = 0.99
        
class place_holder():
    def __init__(self, config):
        self.learning_rate = tf.placeholder(tf.float32)
        self.lres_gt = tf.placeholder(tf.float32, shape = config.lres_dim)
        self.evf_gt = tf.placeholder(tf.float32, shape = config.evf_dim)

def main():
    num_pairs = int(sys.argv[1])
    samp_dim1 = int(sys.argv[2])
    samp_dim2 = int(sys.argv[3])
    samp_xy_dim = (samp_dim1, samp_dim2)
    save_dir = sys.argv[4]
    # prepare samples
    data_dir = '/data/NfS/'
    list_dir = 'need4speed/'
    class_list = os.listdir(data_dir)
    # (train_list, test_list) = split_train_test(class_list)
    # np.save(os.path.join(save_dir, 'partition_list.npy'), (train_list, test_list))
    (train_list, test_list) = np.load(os.path.join(list_dir, 'partition_list.npy'))
    print("train class #", len(train_list))
    print("test class #", len(test_list))

    train_pair_list = pairname_list(train_list, num_pairs)

    X = np.empty((num_pairs, samp_dim1, samp_dim2, 2))
    
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth=True
    
    for isamp in range(num_pairs):
        time_start = time.time()
        hres_gt, lres_gt, evf_gt = prepare_hres_evf(train_list, samp_xy_dim)
        print(np.type(hres_gt))
        print(np.shape(hres_gt))
        time_crop = time.time()
        res_config = config(samp_xy_dim)
        res_ph = place_holder(res_config)

        hres_tensor = init_hres(res_config, lres_gt)
        #flow_tensor = init_flow(res_config)
        lres_tensor = frame_model(res_config, hres_tensor)
        evf_tensor = event_model(res_config, hres_tensor)
        loss = loss_no_flow(res_config, res_ph, hres_tensor, lres_tensor, evf_tensor)

        optimizer = tf.train.AdamOptimizer(learning_rate = res_ph.learning_rate, beta1 = res_config.beta1, beta2 = res_config.beta2)
        opt_min = optimizer.minimize(loss)

        with tf.Session(config = tf_config) as sess:
            sess.run(tf.global_variables_initializer())
            for iepoch in range(res_config.epochs):
                hres, iloss, _ = sess.run([hres_tensor, loss, opt_min],
                                           feed_dict={res_ph.lres_gt: lres_gt, 
                                                      res_ph.evf_gt: evf_gt, 
                                                      res_ph.learning_rate: res_config.lr})
        print("Preparing sample #%s, epoch #%s, crop time: %05f, opt time: %05f" % (isamp, res_config.epochs, time_crop-time_start, time.time()-time_crop))
        pair_1 = np.expand_dims(hres_gt[:,:,:,1], 3)
        pair_2 = np.expand_dims(hres[:,:,:,1], 3)
        X[isamp,] = np.concatenate((pair_1, pair_2), 3)
    # save file
    basename = os.path.join(save_dir, 'training_pairs_%s_%s_%s' % (num_pairs, samp_dim1, samp_dim2))
    basename_ = basename + '.npy'
    num = 0
    while True:
        if os.path.exists(basename_):
            basename_ = basename + '_%04d.npy' % num
            num = num + 1
        else:
            np.save(basename_, X)
            break

if __name__ == '__main__':
    main()
