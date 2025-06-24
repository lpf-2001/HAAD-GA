import os
import sys
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"..")))
from utils.data import load_rimmer_dataset
from Ant_algorithm import HAAD
from DLWF_pytorch.model.model_1000 import VarCNN
import pdb
torch.cuda.empty_cache()
torch.cuda.ipc_collect()

model = VarCNN((5000,5000),100).to('cuda')
model.load_state_dict(torch.load("/root/autodl-tmp/HAAD-GA/DLWF_pytorch/trained_model/length_1000/varcnn_1000.pth"))
X_train, y_train, X_valid, y_valid, X_test, y_test = load_rimmer_dataset(input_size=1000, num_classes=100) #(124991, 1000, 1),(124991, 100)


# pdb.set_trace()


# 启动搜索
haad = HAAD(
    model=model,
    original_trace=X_train,
    numant=10,
    max_inject=5,  # 最多注入 5 个虚包
    max_iter=10
)

best_combination, best_score = haad.run(label=y_train)
print("最优注入位置组合：", best_combination)