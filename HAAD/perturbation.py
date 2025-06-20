import numpy as np
import pygad
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Subset
import json
import sys
import time
import argparse
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),'../'))
sys.path.append(parent_dir)
from utils.data import *
from Ant_algorithm import *
from DLWF_pytorch.model import *
from torch.utils.data import DataLoader,Subset
from tqdm import tqdm
import os
import argparse
import matplotlib.pyplot as plt
from GA import GA_UniversalPatch




parser = argparse.ArgumentParser(description='Train and test a deep neural network (SDAE, CNN or LSTM)')


torch.cuda.empty_cache()
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


print("=============Parameter Settings=============")

parser.add_argument('--s_model', '-sm', default='ensemble', type=str, choices=['cnn','lstm','sdae','varcnn','df','ensemble'],help='choose model type cnn, lstm or sdae')

parser.add_argument('--v_model', '-vm', default='df', type=str, choices=['cnn','lstm','sdae','varcnn','df','ensemble'],help='choose model type cnn, lstm or sdae')
parser.add_argument('--patch_start', '-ps', default=8, type=int)


parser.add_argument('--patch_end', '-pe', default=9, type=int)
args = parser.parse_args()


max_insert = 6
numant = 4
itermax = 10
val_ratio = 0.7
test_ratio = 0.2


for patch in range(args.patch_start,args.patch_end):
    # 加载数据集
    X_train, y_train, X_valid, y_valid, X_test, y_test=load_rimmer_dataset(input_size=200,test_ratio=test_ratio,val_ratio=val_ratio)
    print("val_ratio:",val_ratio,"test_ratio:",test_ratio)
    print("X_valid samples",X_valid.shape[0],"X_test samples",X_test.shape[0])
    print("Use X_train shape:",X_train.shape)
    train_dataset = MyDataset(X_train,y_train)
    val_dataset = MyDataset(X_valid,y_valid)
    test_dataset = MyDataset(X_test,y_test)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)
    UniversalPatch = GA_UniversalPatch(patches=patch,s_model=args.s_model,train_loader=train_loader,max_insert=max_insert,numant=numant,itermax=itermax)
    x,y = UniversalPatch.run()
    print(x)
    np.save(f'./result/{patch}patch_{args.s_model}.npy',x)
    np.save(f'./result/{patch}patch_{sum(x[:,1])}_{args.s_model}.npy',x)
    