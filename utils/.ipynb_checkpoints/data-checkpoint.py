import numpy as np
from keras.utils import to_categorical
from collections import Counter
from torch.utils.data import DataLoader, Dataset
import tensorflow.keras.preprocessing.sequence as sq
# 打印 TensorFlow 和 Keras 版本
import os 
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn.functional as F


current_dir = os.path.dirname(os.path.abspath(__file__))
print("data.py working directory:", current_dir)

def format_data(X, y, input_size, num_classes):
    """
    Format traces into input shape [N x Length x 1] and one-hot encode labels.
    """
    X = X[:, :input_size]
    X = X.astype('float32')
    X = X[:, :, np.newaxis]

    y = y.astype('int32')
    y = np.eye(num_classes)[y]

    return X, y


def format_data_all(X_train, y_train, X_valid, y_valid, X_test, y_test, input_size, num_classes):
    X_train, y_train = format_data(X_train, y_train, input_size, num_classes)
    X_valid, y_valid = format_data(X_valid, y_valid, input_size, num_classes)
    X_test, y_test = format_data(X_test, y_test, input_size, num_classes)

    return X_train, y_train, X_valid, y_valid, X_test, y_test
def train_test_valid_split(X, y, valid_size=0.1, test_size=0.1):
    """
    Split data into training, validation, and test sets.
    Set random_state=0 to keep the same split.
    """
    # Split into training set and others
    split_size = valid_size + test_size
    [X_train, X_, y_train, y_] = train_test_split(X, y,
                                    test_size=split_size,
                                    random_state=0,
                                    stratify=y)

    # Split into validation set and test set
    split_size = test_size / (valid_size + test_size)
    [X_valid, X_test, y_valid, y_test] = train_test_split(X_, y_,
                                            test_size=split_size,
                                            random_state=0,
                                            stratify=y_)

    return X_train, y_train, X_valid, y_valid, X_test, y_test

def LoadSirinam(input_size, num_classes,formatting=True,val_ratio=0.25,test_ratio=0.25,closed_world=True):
   
    print("Loading non-defended dataset for closed-world scenario")
    # Point to the directory storing data
    dataset_dir = current_dir + "/Dataset/Sirinam/"

    # X represents a sequence of traffic directions
    # y represents a sequence of corresponding label (website's label)
    if closed_world:
        dataset_dir = dataset_dir + 'Closed_world/'
        # Load training data
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
    else:
        dataset_dir = dataset_dir + 'Open_world/'
        # Load training data
        with open(dataset_dir + 'X_train_NoDef-2.pkl', 'rb') as handle:
            
            X_train = np.array(pickle.load(handle , encoding='bytes'))
        with open(dataset_dir + 'y_train_NoDef-2.pkl', 'rb') as handle:
            y_train = np.array(pickle.load(handle, encoding='bytes'))

        # Load validation data
        with open(dataset_dir + 'X_test_Mon_NoDef.pkl', 'rb') as handle:
            X_valid = np.array(pickle.load(handle, encoding='bytes'))
        with open(dataset_dir + 'y_test_Mon_NoDef.pkl', 'rb') as handle:
            y_valid = np.array(pickle.load(handle, encoding='bytes'))

        # Load testing data
        with open(dataset_dir + 'X_valid_NoDef-2.pkl', 'rb') as handle:
            X_test = np.array(pickle.load(handle, encoding='bytes'))
        with open(dataset_dir + 'y_valid_NoDef-2.pkl', 'rb') as handle:
            y_test = np.array(pickle.load(handle, encoding='bytes'))

    print( "Data dimensions:")
    print ("X: Training data's shape : ", X_train.shape)
    print ("y: Training data's shape : ", y_train.shape)
    print ("X: Validation data's shape : ", X_valid.shape)
    print ("y: Validation data's shape : ", y_valid.shape)
    print ("X: Testing data's shape : ", X_test.shape)
    print ("y: Testing data's shape : ", y_test.shape)
    all_x = np.concatenate((X_train,X_valid,X_test),axis=0)
    all_y = np.concatenate((y_train,y_valid,y_test),axis=0)
    category_count = Counter(all_y)
    count_ = 0
    for category, count in category_count.items():
        print(f"类别 {category} 的数量是: {count}")
        count_ += 1
        if count_ == 4:
            break
    X_train, y_train, X_valid, y_valid, X_test, y_test = train_test_valid_split(all_x,all_y, valid_size=val_ratio, test_size=test_ratio)
    if formatting:
        return format_data_all(X_train, y_train, X_valid, y_valid, X_test, y_test, input_size, num_classes)
    else:
        return X_train, y_train, X_valid, y_valid, X_test, y_test


def load_rimmer_dataset(input_size=5000, num_classes=100, formatting=True,test_ratio=0.25,val_ratio = 0.25):
    """
    Load Rimmer's (NDSS'18) dataset.
    """
    # Point to the directory storing data
    dataset_dir = current_dir+"/Dataset/Rimmer/"
    # datafile = '../Dataset/tor_100w_2500tr.npz'

    # Load data
    datafile = dataset_dir + 'tor_%dw_2500tr.npz' % num_classes
    with np.load(datafile, allow_pickle=True) as npzdata:
        data = npzdata['data']
        labels = npzdata['labels']
    print("-----------")
    # Convert website to integer

    websites, y = np.unique(labels,return_inverse=True)
    
    print(y)
    # Split data to fixed parts
    X_train, y_train, X_valid, y_valid, X_test, y_test = train_test_valid_split(data, y, valid_size=val_ratio, test_size=test_ratio)
    # with open(dataset_dir + 'tor_%dw_2500tr_test.npz' % num_classes, 'wb') as handle:
    #     pickle.dump({'X_test': X_test, 'y_test': y_test}, handle)

    if formatting:
        return format_data_all(X_train, y_train, X_valid, y_valid, X_test, y_test, input_size, num_classes)
    else:
        return X_train, y_train, X_valid, y_valid, X_test, y_test
    


def train_test_valid_split(X, y, valid_size=0.1, test_size=0.1):
    """
    Split data into training, validation, and test sets.
    Set random_state=0 to keep the same split.
    """
    # Split into training set and others
    split_size = valid_size + test_size
    [X_train, X_, y_train, y_] = train_test_split(X, y,
                                    test_size=split_size,
                                    random_state=0,
                                    stratify=y)

    # Split into validation set and test set
    split_size = test_size / (valid_size + test_size)
    [X_valid, X_test, y_valid, y_test] = train_test_split(X_, y_,
                                            test_size=split_size,
                                            random_state=0,
                                            stratify=y_)

    return X_train, y_train, X_valid, y_valid, X_test, y_test


class MyDataset(Dataset):
    def __init__(self,data,labels):
        # print(data.dtype)
        self.data = torch.from_numpy(data).float()
        self.labels = torch.from_numpy(labels).long()
    def __len__(self):
        return len(self.data)
    def __getitem__(self,idx):
    
        return self.data[idx], self.labels[idx]