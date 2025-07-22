import torch
from model_1000 import VarCNN
import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),'../'))
from utils.data import load_rimmer_dataset

model = VarCNN((1000,100),100).to('cuda')
model.load_state_dict(torch.load('/root/autodl-tmp/HAAD-GA/DLWF_pytorch/trained_model/length_1000/varcnn_1000.pth'))
x_train,y_train,x_valid,y_valid,x_test,y_test = load_rimmer_dataset(1000,100)
x_train = torch.tensor(x_train).to('cuda')
y_train = torch.tensor(y_train).to('cuda')
batch_size = 64
for i in range(0,x_train.shape[0],batch_size):
    y_pre = model(x_train[i:i+batch_size,:,:])
    print(y_pre.argmax(-1)[i:i+batch_size])
    print(y_train.argmax(-1)[i:i+batch_size])
    break