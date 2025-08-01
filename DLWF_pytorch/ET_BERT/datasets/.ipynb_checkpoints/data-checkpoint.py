import numpy as np
import os 
import pickle
import torch

from sklearn.model_selection import train_test_split

current_dir = os.path.dirname(os.path.abspath(__file__))
print("data.py working directory:", current_dir)


def read_dataset(args, type_="Rimmer"):

    if type_ == "Sirinam":
    
        dataset_dir = current_dir+'/Sirinam/'
        with open(dataset_dir + 'X_train_NoDef.pkl', 'rb') as handle:     
            X_train = np.array(pickle.load(handle , encoding='bytes'))
        with open(dataset_dir + 'y_train_NoDef.pkl', 'rb') as handle:
            y_train = np.array(pickle.load(handle, encoding='bytes'))

        # Load validation data
        with open(dataset_dir + 'X_valid_NoDef.pkl', 'rb') as handle:
            X_valid = np.array(pickle.load(handle, encoding='bytes'))
        with open(dataset_dir + 'y_valid_NoDef.pkl', 'rb') as handle:
            y_valid = np.array(pickle.load(handle, encoding='bytes'))

        # Load testing data
        with open(dataset_dir + 'X_test_NoDef.pkl', 'rb') as handle:
            X_test = np.array(pickle.load(handle, encoding='bytes'))
        with open(dataset_dir + 'y_test_NoDef.pkl', 'rb') as handle:
            y_test = np.array(pickle.load(handle, encoding='bytes'))
            
        data = np.concatenate((X_train,X_valid,X_test),axis=0)
        y = np.concatenate((y_train,y_valid,y_test),axis=0)
    elif type_ == "Rimmer":
        datafile = current_dir+'/../../../utils/Dataset/Rimmer/tor_100w_2500tr.npz'
        with np.load(datafile, allow_pickle=True) as npzdata:
            data = npzdata['data']
            labels = npzdata['labels']
        # Convert website to integer
        y = labels.copy()
        websites = np.unique(labels)
        for w in websites:
            y[np.where(labels == w)] = np.where(websites == w)[0][0]

    X_train, X_, y_train, y_ = train_test_split(data, y, test_size=0.9,random_state=0,stratify=y)

    
    return X_train,y_train,X_,y_