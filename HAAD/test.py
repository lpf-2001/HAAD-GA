import os
import sys
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"..")))
from utils.data import load_rimmer_dataset
from Ant_algorithm import Ant
from DLWF_pytorch.model.model_1000 import VarCNN
torch.cuda.empty_cache()
torch.cuda.ipc_collect()

model = VarCNN((5000,5000),100).to('cuda')
model.load_state_dict(torch.load("/root/autodl-tmp/HAAD-GA/DLWF_pytorch/trained_model/length_1000/varcnn_1000.pth"))
X_train, y_train, X_valid, y_valid, X_test, y_test = load_rimmer_dataset(input_size=1000, num_classes=100)
haad = Ant(model,10,10,10,10)
print(X_train.shape)
haad.sensitive_results(X_train[:5000,:,:],y_train[:5000])