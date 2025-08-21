import os
import sys
import torch
import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"..")))

sys.path.append('/root/workspace/damodel/HAAD_GA/DLWF_pytorch/ET_BERT')

print(sys.path)
from utils.data import load_rimmer_dataset
from HAAD.Ant_algorithm import HAAD
from DLWF_pytorch.model.model_5000 import *
from utils.data import load_rimmer_dataset
from sklearn.model_selection import train_test_split
import pdb
work_dir = os.path.dirname(os.path.abspath(__file__))
torch.cuda.empty_cache()
torch.cuda.ipc_collect()

num_classes = 100

def print_gpu_memory():
    allocated = torch.cuda.memory_allocated() / 1024 ** 2  # 当前分配的显存
    reserved = torch.cuda.memory_reserved() / 1024 ** 2    # 当前保留的显存（包括缓存）
    max_allocated = torch.cuda.max_memory_allocated() / 1024 ** 2  # 历史最大分配
    max_reserved = torch.cuda.max_memory_reserved() / 1024 ** 2    # 历史最大保留
    print(f"[GPU 显存] 当前分配: {allocated:.2f} MB，当前保留: {reserved:.2f} MB，"
          f"最大分配: {max_allocated:.2f} MB，最大保留: {max_reserved:.2f} MB")





parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

args = parser.parse_args()

args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model = Classifier(args)
# model = load_model(model, args.load_model_path).to(args.device)
model = MultiScaleLLM_V2(num_classes=num_classes).to(device)
model.load_state_dict(torch.load(work_dir+'/../DLWF_pytorch/trained_model/length_5000/Rimmer/llm_rimmer100.pth'))
print_gpu_memory()


model2 = VarCNN(5000,100).to(device)
model2.load_state_dict(torch.load(work_dir+'/../DLWF_pytorch/trained_model/length_5000/Rimmer/varcnn_rimmer100.pth'))
# model2 = DFNet(100).to(args.device)
# model2.load_state_dict(torch.load(work_dir+'/../DLWF_pytorch/trained_model/length_5000/df_5000.pth'))
X_train, y_train, X_valid, y_valid, X_test, y_test = load_rimmer_dataset(input_size=5000, num_classes=num_classes) #(124991, 1000, 1),(124991, 100)


# pdb.set_trace()
# 启动搜索
haad = HAAD(
    model=model,
    original_trace=X_train,
    numant=10,
    max_inject=500,  # 最多注入 5 个虚包
    max_iter=1
)
best_combination, best_score = haad.run(labels=y_train)
loss,correct = haad.evaluate_solution(model=model2,labels=y_test,original_trace=X_test,solution=best_combination)
print("最优注入位置组合：", best_combination)
print("预测准确率：",correct/y_train.shape[0])