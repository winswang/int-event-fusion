import numpy as np
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_psnr as psnr

def computePsnrSsim(hres_gt, hres_rec):
    # input hres_gt (t, y, x, c)
    # input hres_rec (1, t, y, x, c)
    frameNum = np.size(hres_rec, 1)
    psnr_vec = []
    ssim_vec = []
    for i in range(frameNum):
        if i >= 0:
            if i <= frameNum - 1:
                psnr_vec.append(psnr(hres_gt[i], hres_rec[0][i]))
                ssim_vec.append(ssim(hres_gt[i], hres_rec[0][i], multichannel = True))
    return psnr_vec, ssim_vec

def evfToFloatImg(evf, mode = 'm'):
    # input will be in shape (1, y, x, n)
    if np.size(evf, 3) == 0:
        evf = np.expand_dims(evf)
    (dim0, dim1, dim2, dim3) = np.shape(evf)
    if mode == 'm':
        eMin = np.min(evf)
        eMax = np.max(evf)
    else:
        eMin = -1
        eMax = 1
    fac = eMax - eMin
    img = np.empty((dim1, dim2, dim3))
    for i in range(dim3):
        img[:,:,i] = (evf[0,:,:,i] - eMin)/fac
    return img

def imgNonNeg(img):
    img[img < 0] = 0
    return img

def imgBetween01(img):
    img[img < 0] = 0
    img[img > 1] = 1
    return img

def plot_color_evf(evf):
    # evf should in shape (y, x, n)
    dim1 = np.size(evf, 0)
    dim2 = np.size(evf, 1)
    neg_c = np.array([212, 20, 90])/255.
    pos_c = np.array([63, 169, 245])/255.
    canvas = np.ones((dim1, dim2, 3))*0.5
    for i in range(dim1):
        for j in range(dim2):
            if evf[i,j] == 1:
                canvas[i,j,:] = pos_c
            elif evf[i,j] == -1:
                canvas[i,j,:] = neg_c
    return canvas

def vid2evf(vid, event_thres = 0.08):
    dim0, dim1, dim2 = np.shape(vid)
    lvid = np.log(vid+1e-10)
    evf = np.zeros((dim0-1, dim1, dim2))
    for i in range(dim0 - 1):
        frame_diff = lvid[i+1,:,:] - lvid[i,:,:]
        ievf = np.zeros_like(frame_diff,dtype=np.float)
        ievf[frame_diff>event_thres] = 1.0
        ievf[frame_diff<-event_thres] = -1.0
        if i == 0:
            evf = ievf
        else:
            evf = np.dstack((evf, ievf))
    return evf

def norm_per_frame(vid):
    return np.array([norm_max(vid[i,:,:]) for i in range(np.size(vid,0))])

def norm_max(x):
    return x/np.amax(x)