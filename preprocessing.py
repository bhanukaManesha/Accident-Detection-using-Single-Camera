from numpy import load
import os
import numpy as np
import re

def npy2numpy(file_path):
    return load(file_path)

def npz2npy(npz_file_path,new_file_path):
    npz_file = load(npz_file_path)
    batchno = re.sub("\D", "", npz_file_path)
    for item in npz_file:
        inner_item = npz_file[item]
        np.save(new_file_path + item + batchno + '.npy', inner_item)
        
def batchnpz2npy(file_path):
    files = os.listdir(file_path)
    files = sorted(files)
    for item in files:
        npz2npy(file_path+item, file_path+"ext/")
        print("Extracting " + item + " done")

def sep_frame_object(data_object):
        full_feature = data_object[:,:,0,:]
        object_features = data_object[:,:,1:,:]
        # combine_frame_object(full_feature,object_features)

def combine_frame_object(full_feature,object_features):
        full_feature = np.expand_dims(full_feature, axis=2)
        data = np.concatenate((full_feature,object_features), axis=2)
        return data


if __name__ == "__main__":
    n = load('train/training_features/data001.npy')
    sep_frame_object(n)
    

