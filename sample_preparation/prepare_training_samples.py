import numpy as np
import tensorflow as tf
import os, sys
import argparse
import random, json
#from keras.preprocessing.image import img_to_array, load_img
from skimage.io import imread
import matplotlib.pyplot as plt
#from keras.layers import Input, Conv2D, BatchNormalization, Activation, Subtract
#from keras.models import Model, load_model
#from keras.optimizers import Adam
#import keras.backend as K
import time

# class of load clip
class load_videos():
    def __init__(self, info_file = 'adobe-nfs-split.txt', syn_mode = 'p1', channel = 3,
                 num_videos = None, dim_yx = (40, 40), train_test = 1, verbose = 0):
        self.info_file = info_file
        # parse syn_mode
        self.mode = syn_mode[0]
        self.unknown = int(syn_mode[1:])
        # generate hres_gt, lres_gt, evf_gt
        if self.mode == 'p':
            self.read_len = self.unknown + 1
        elif self.mode == 'i':
            self.read_len = self.unknown + 2
        elif self.mode == 'm':
            self.read_len = self.unknown

        self.load_split_file(self.info_file)
        self.channel = channel
        self.num_videos = num_videos
        self.dim_y, self.dim_x = dim_yx
        self.train_test = train_test
        self.read_idx_list = self.rand_video_idx()

        self.evth_max = 0.2
        self.verbose = verbose

    def rand_video_idx(self):
        read_list = [i for i, x in enumerate(self.train_test_list) if x == self.train_test]
        return random.sample(read_list, self.num_videos)
    def load_split_file(self, filename, shuffle=False):
        self.images_list = []
        self.start_frame = []
        self.img_dim_y_list = []
        self.img_dim_x_list = []
        self.train_test_list = []
        with open(filename) as f:
            lines_list = f.readlines()
            if shuffle:
                random.shuffle(lines_list)

            for lines in lines_list:
                line = lines.rstrip('\n').split(' ')
                if line[0] == 'train':
                    self.train_test_list.append(int(1))
                else:
                    self.train_test_list.append(int(0))
                self.images_list.append(line[1])
                self.img_dim_y_list.append(line[2])
                self.img_dim_x_list.append(line[3])
                self.start_frame.append(int(line[4]))
        return True

    def read_image(self, filename, normalization=True):
        if self.channel == 1:
            image = np.array(imread(filename, as_gray = True))
            image = np.expand_dims(image[self.y:(self.y + self.dim_y), self.x:(self.x + self.dim_x)],
                                   axis = 2)
        else:
            image = np.array(imread(filename))
            image = image[self.y:(self.y + self.dim_y), self.x:(self.x + self.dim_x),:]

        if normalization:
            image = image / 255.0
        if self.flip1 > 0.5:
            image = np.fliplr(image)
        if self.flip2 > 0.5:
            image = np.flipud(image)
        if self.rot90 < 0.25:
            image = np.rot90(image, 1)
        elif self.rot90 < 0.5:
            image = np.rot90(image, 2)
        elif self.rot90 < 0.75:
            image = np.rot90(image, 3)
        return image

    def load_clip(self, img_dir, start_idx, frame_num, verbose = 0):
        self.y = random.randint(0, self.img_dim_y-self.dim_y-1)
        self.x = random.randint(0, self.img_dim_x-self.dim_x-1)
        self.flip1 = random.random()
        self.flip2 = random.random()
        self.rot90 = random.random()
        if 'Adobe' in img_dir:
            vid = np.array([self.read_image(os.path.join(img_dir, '%04d.jpg' % (start_idx + i))) for i in range(frame_num)])
        else:
            vid = np.array([self.read_image(os.path.join(img_dir, '%05d.jpg' % (start_idx + i))) for i in range(frame_num)])

        if verbose:
            print("-- Finished loading clip, shape of",np.shape(vid))
        return vid

    def cvid2evf(self, vid):
        # convert color video to event frames
        # (t, y, x, c)
        dim0, dim1, dim2, dim3 = np.shape(vid)
        evfFrameNum = dim0 - 1
        event_thres = random.uniform(0.01, self.evth_max)
        lvid = np.log(vid / 255.0 + 1e-10)
        evf = np.zeros((evfFrameNum, dim1, dim2, dim3))
        for i in range(evfFrameNum):
            frame_diff = lvid[i + 1] - lvid[i]
            ievf = np.zeros(np.shape(frame_diff))
            ievf[frame_diff > event_thres] = 1.0
            ievf[frame_diff < -event_thres] = -1.0
            evf[i] = ievf
        return evf
    def generate_hle(self, vid_idx):
        # read one set of hres， lres, evf
        read_dir = self.images_list[vid_idx]
        read_start_idx = self.start_frame[vid_idx]
        self.img_dim_x = int(self.img_dim_x_list[vid_idx])
        self.img_dim_y = int(self.img_dim_y_list[vid_idx])
        hres_gt = self.load_clip(read_dir, read_start_idx, self.read_len, self.verbose)
        evf_gt = self.cvid2evf(hres_gt)
        if self.mode == 'i':
            lres_gt = np.array([hres_gt[0], hres_gt[-1]])
        elif self.mode == 'p':
            lres_gt = np.array([hres_gt[0]])
        elif self.mode == 'm':
            lres_gt = np.expand_dims(np.mean(hres_gt, axis = 0), axis = 0)
        hres_dim = np.shape(hres_gt)
        lres_dim = np.shape(lres_gt)
        evf_dim = np.shape(evf_gt)
        if self.verbose == 1:
            print("-- Finished generating hle:", hres_dim, lres_dim, evf_dim)
        return hres_gt, lres_gt, evf_gt, hres_dim, lres_dim, evf_dim

    def generate_sample_pairs(self):
        for ipair in range(self.num_videos):
            hres_gt, lres_gt, evf_gt, hres_dim, lres_dim, evf_dim = self.generate_hle(ipair)
            dmr_config = rec_config(hres_dim= hres_dim, lres_dim= lres_dim, evf_dim= evf_dim,
                                    syn_mode=self.mode, unknown_t=self.unknown, c_channel=self.channel,
                                    verbose=self.verbose)
            dmr_ph = place_holder(dmr_config)
            hres_tensor = init_hres(dmr_config, lres_gt)
            lres_tensor = frame_model(dmr_config, hres_tensor)
            evf_tensor = event_model(dmr_config, hres_tensor)
            loss = loss_pix_tv(dmr_config, dmr_ph, hres_tensor, lres_tensor, evf_tensor)
            optimizer = tf.train.AdamOptimizer(learning_rate=dmr_ph.learning_rate, beta1=dmr_config.beta1,
                                               beta2=dmr_config.beta2)
            opt_min = optimizer.minimize(loss)
            sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
            sess.run(tf.global_variables_initializer())
            for iepoch in range(dmr_config.epochs):
                if iepoch == 0:
                    dmr_config.lr = dmr_config.lr_init
                hres_rec, _, _ = sess.run([hres_tensor, loss, opt_min],
                                                      feed_dict={dmr_ph.lres_gt: lres_gt,
                                                                 dmr_ph.evf_gt: evf_gt,
                                                                 dmr_ph.learning_rate: dmr_config.lr})
                if iepoch % dmr_config.lr_update == 0:
                    if iepoch > 0:
                        dmr_config.lr *= 0.1
            hres_gt = np.expand_dims(hres_gt, axis = 0)
            Xpair = np.transpose(np.concatenate((hres_gt, hres_rec), axis=0), (1, 0, 2, 3, 4))
            if ipair == 0:
                X = Xpair
            else:
                print(np.shape(X))
                print(np.shape(Xpair))
                X = np.concatenate((X, Xpair), axis = 0)
            if self.verbose:
                print("-Finished preparing #", ipair)
        return X


# config
class rec_config():
    def __init__(self, hres_dim=None, lres_dim=None, evf_dim=None, syn_mode = 'i', unknown_t=None,
                 c_channel = 3, verbose = 0):
        self.syn_mode = syn_mode
        self.hres_dim = hres_dim
        self.unknown = unknown_t
        self.lres_dim = lres_dim
        self.evf_dim = evf_dim
        self.c_channel = c_channel
        self.verbose = verbose

        # events
        self.ev_weight = random.uniform(3e-1, 10e-1)#6e-1
        self.tanh_coef = random.uniform(5.0, 10.0)#7.0

        # hres_tv
        self.tv_coef_xy = random.uniform(1e-2, 1e-1)#5e-2
        self.tv_coef_t = random.uniform(5e-2,5e-1)#1e-1

        # learning
        self.lr_init = random.uniform(2e-3, 20e-3)#20e-3
        self.lr_update = 100
        self.epochs = random.randint(1, 300)
        self.beta1 = 0.9
        self.beta2 = 0.99

class place_holder():
    def __init__(self, config):
        self.learning_rate = tf.placeholder(tf.float32)
        self.lres_gt = tf.placeholder(tf.float32, shape = config.lres_dim)
        self.evf_gt = tf.placeholder(tf.float32, shape = config.evf_dim)

# models
def init_hres(config, lres_gt):
    # input lres_gt should be in shape (t, y, x, c)
    if config.syn_mode == 'i':
        lres_mean = np.expand_dims(np.mean(lres_gt, axis = 0), axis = 0)
        #print("shape of 'lres_mean'", np.shape(lres_mean))
        # (1, y, x, c)
        lres_start = tf.Variable(np.expand_dims(lres_gt[0], axis = 0), dtype = tf.float32)
        lres_end = tf.Variable(np.expand_dims(lres_gt[1], axis = 0), dtype = tf.float32)
        # init hres
        for i in range(config.unknown+2):
            a = i / (config.unknown + 1)
            if i == 0:
                hres_init = lres_start
            else:
                frame = (1-a)*lres_start + a*lres_end
                hres_init = tf.concat([hres_init, tf.Variable(frame, dtype = tf.float32)], 0)
        hres_init = tf.expand_dims(hres_init, axis = 0)
    elif config.syn_mode == 'p':
        for i in range(config.unknown+1):
            if i == 0:
                hres_init = tf.Variable(lres_gt, dtype= tf.float32)
            else:
                hres_init = tf.concat([hres_init, tf.Variable(lres_gt, dtype= tf.float32)], 0)
        hres_init = tf.expand_dims(hres_init, axis = 0)
    else:
        for i in range(config.unknown):
            if i == 0:
                hres_init = tf.Variable(lres_gt, dtype= tf.float32)
            else:
                hres_init = tf.concat([hres_init, tf.Variable(lres_gt, dtype= tf.float32)], 0)
        hres_init = tf.expand_dims(hres_init, axis = 0)
    print("hres_init shape:", np.shape(hres_init))
    return hres_init

def frame_model(config, hres_tensor):
    if config.syn_mode == 'i':
        lres_tensor = tf.transpose(tf.gather_nd(tf.transpose(hres_tensor, perm = [1,2,3,4,0]),
                                                indices = [[0],[config.unknown+1]]),
                                   perm = [4,0,1,2,3])
    elif config.syn_mode == 'p':
        lres_tensor = tf.transpose(tf.gather_nd(tf.transpose(hres_tensor, perm=[1,2,3,4,0]),
                                                indices = [[0]]), perm=[4,0,1,2,3])
    else:
        lres_tensor = tf.reduce_sum(hres_tensor, 1, keepdims = True)
    if config.verbose == 1:
        print("----Finished generating lres_tensor, shape", np.shape(lres_tensor))
    return lres_tensor

def event_model(config, hres_tensor):
    hres_relu = tf.nn.relu(hres_tensor)
    tanh_coef = tf.constant(config.tanh_coef)
    kernel = tf.constant([-1, 1], dtype=tf.float32)
    k_tyx = tf.expand_dims(tf.expand_dims(tf.expand_dims(kernel, axis=1), axis=2), axis=3)
    if config.c_channel == 3:
        k_tyxi = tf.expand_dims(tf.concat([k_tyx, k_tyx, k_tyx], axis=3), axis=4)
        k_tyxio = tf.concat([k_tyxi, k_tyxi, k_tyxi], axis=4)
    else:
        k_tyxio = tf.expand_dims(k_tyx, axis = 4)
    evf_tanh = tf.tanh(
        tanh_coef * tf.nn.convolution(input=hres_relu, filter=k_tyxio, padding="VALID", data_format="NDHWC"))
    if config.verbose == 1:
        print("evf_tanh, shape:", np.shape(evf_tanh))
    return evf_tanh

def tv_2d(config, hres_tensor):
    kernel = tf.constant([-1, 1], dtype=tf.float32)
    kx = tf.expand_dims(tf.expand_dims(tf.expand_dims(kernel, axis=0), axis=2), axis=3)
    ky = tf.expand_dims(tf.expand_dims(tf.expand_dims(kernel, axis=1), axis=2), axis=3)
    if config.c_channel == 3:
        kxi = tf.expand_dims(tf.concat([kx, kx, kx], axis=3), axis=4)
        kxio = tf.concat([kxi, kxi, kxi], axis=4)
        kyi = tf.expand_dims(tf.concat([ky, ky, ky], axis=3), axis=4)
        kyio = tf.concat([kyi, kyi, kyi], axis=4)
    else:
        kxio = tf.expand_dims(kx, axis = 4)
        kyio = tf.expand_dims(ky, axis = 4)

    dx = tf.nn.convolution(input=hres_tensor, filter=kxio, padding="SAME", data_format="NDHWC")
    dy = tf.nn.convolution(input=hres_tensor, filter=kyio, padding="SAME", data_format="NDHWC")
    return tf.norm(dx + dy, ord=1)

def tv_t(config, hres_tensor):
    kernel = tf.constant([-1, 1], dtype=tf.float32)
    kt = tf.expand_dims(tf.expand_dims(tf.expand_dims(kernel, axis=0), axis=0), axis=3)
    if config.c_channel == 3:
        kti = tf.expand_dims(tf.concat([kt, kt, kt], axis=3), axis=4)
        ktio = tf.concat([kti, kti, kti], axis=4)
    else:
        ktio = tf.expand_dims(kt, axis = 4)
    dt = tf.nn.convolution(input=hres_tensor, filter=ktio, padding="VALID", data_format="NDHWC")
    return tf.norm(dt, ord=1)

def loss_pix_tv(config, ph, hres_tensor, lres_tensor, evf_tensor):
    frame_loss = tf.norm(ph.lres_gt - tf.squeeze(lres_tensor, [0]), ord = 1)
    event_loss = tf.constant(config.ev_weight)*tf.norm(ph.evf_gt - tf.squeeze(evf_tensor, [0]), ord = 1)
    tv_loss = tf.constant(config.tv_coef_xy)*tv_2d(config, hres_tensor) + tf.constant(config.tv_coef_t)*tv_t(config, hres_tensor)
    return frame_loss + event_loss + tv_loss


def main():
    # default values:
    syn_mode = 'p1'
    num_epoch = 200
    save_dir = 'F:\Winston\data\clips-in-npy'
    num_channels = 3
    num_videos = 233
    info_file = 'adobe-nfs-split.txt'
    verbose = 0

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", help="synthetic mode: i-interp; p-predict; m-motion deblur;"
                                             " + unknown frame #")
    parser.add_argument("-e", "--num_epochs", help="number of epochs", type=int)
    parser.add_argument("--save_dir", help="save directory")
    parser.add_argument("--info_file", help="path to adobe-nfs-split.txt file")
    parser.add_argument("-c", "--color", help="number of color channels", type=int)
    parser.add_argument("--clips", help="number of clips", type=int)
    parser.add_argument("-v", "--verbose", help="verbosity", type=int)
    args = parser.parse_args()
    if args.mode:
        syn_mode = args.mode
    if args.num_epochs:
        num_epoch = args.num_epochs
    if args.save_dir:
        save_dir = args.save_dir
    if args.info_file:
        info_file = args.info_file
    if args.color:
        num_channels = args.color
    if args.clips:
        num_videos = args.clips
    if args.verbose:
        verbose = args.verbose

    if verbose:
        print("* Synthetic mode:", syn_mode)
        print("* # of epochs:", num_epoch)
        print("* # of clips:", num_videos)
        print("* # of channels:", num_channels)
        print("* save to %s" % save_dir)
    # create sub-directory if not exist
    train_mode_dir = os.path.join(save_dir, 'train', '%s' % syn_mode)
    test_mode_dir = os.path.join(save_dir, 'test', '%s' % syn_mode)
    if not os.path.isdir(train_mode_dir):
        os.mkdir(train_mode_dir)
    train_dir = os.path.join(train_mode_dir, 'v%s' % num_videos)
    if not os.path.isdir(train_dir):
        os.mkdir(train_dir)
    if not os.path.isdir(test_mode_dir):
        os.mkdir(test_mode_dir)
    test_dir = os.path.join(test_mode_dir, 'v%s' % num_videos)
    if not os.path.isdir(test_dir):
        os.mkdir(test_dir)

    samp_dim1 = 40
    samp_dim2 = 40
    # prepare samples
    train_info = load_videos(info_file, syn_mode = syn_mode, channel = num_channels, num_videos = num_videos,
                         dim_yx= (samp_dim1, samp_dim2), train_test= 1, verbose= verbose)
    test_info = load_videos(info_file, syn_mode=syn_mode, channel=num_channels, num_videos=int(num_videos*0.25),
                             dim_yx=(samp_dim1, samp_dim2), train_test=0, verbose=verbose)

    save_num = 0
    for iep in range(num_epoch):
        print("Preparing Epoch #", iep+1)
        X_train = train_info.generate_sample_pairs()
        while os.path.exists(os.path.join(train_dir, '%05d.npy' % save_num)):
            save_num += 1
        np.save(os.path.join(train_dir, '%05d.npy' % save_num), X_train)
        X_test = test_info.generate_sample_pairs()
        # first check if file exists
        while os.path.exists(os.path.join(test_dir, '%05d.npy' % save_num)):
            save_num += 1
        np.save(os.path.join(test_dir, '%05d.npy' % save_num), X_test)

if __name__ == '__main__':
    main()
