import glob
import numpy as np

def gen_train_data(data_dir="/data/nfs_training_pairs/", verbose=False):
    filelist = glob.glob(data_dir+'*.npy')
    data = []
    for i in range(len(filelist)):
        data.append(np.load(filelist[i]))
        if verbose:
            print(str(i+1)+'/'+ str(len(filelist)) + ' is done ^_^')
    data = np.array(data)
    data = data.reshape((data.shape[0]*data.shape[1],data.shape[2],data.shape[3],2))
    data_x = data[:int(len(data)*0.8),:,:,1:]
    data_y = data[:int(len(data)*0.8),:,:,:1]
#     discard_n = len(data)-len(data)//args.batch_size*args.batch_size;
#     data = np.delete(data,range(discard_n),axis = 0)
    print('^_^-training data finished-^_^')
    return data_x,data_y


def gen_validate_data(data_dir="/data/nfs_training_pairs/", verbose=False):
    filelist = glob.glob(data_dir+'*.npy')
    data = []
    for i in range(len(filelist)):
        data.append(np.load(filelist[i]))
        if verbose:
            print(str(i+1)+'/'+ str(len(filelist)) + ' is done ^_^')
    data = np.array(data)
    data = data.reshape((data.shape[0]*data.shape[1],data.shape[2],data.shape[3],2))
    data_x = data[int(len(data)*0.8):,:,:,1:]
    data_y = data[int(len(data)*0.8):,:,:,:1]
#     discard_n = len(data)-len(data)//args.batch_size*args.batch_size;
#     data = np.delete(data,range(discard_n),axis = 0)
    print('^_^-validation data finished-^_^')
    return data_x,data_y