import os
import sys
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"..")))
from uer.utils.constants import *
from utils.data import load_rimmer_dataset
from Ant_algorithm import HAAD
from uer.utils import *
from uer.utils.config import load_hyperparam
from uer.utils.seed import set_seed
from uer.model_loader import load_model
from uer.opts import infer_opts
from datasets.data import read_dataset
from fine_tuning.run_classifier import Classifier
from sklearn.model_selection import train_test_split
import argparse
from uer.model_loader import load_model
from uer.opts import infer_opts
import pdb
torch.cuda.empty_cache()
torch.cuda.ipc_collect()

import torch

def print_gpu_memory():
    allocated = torch.cuda.memory_allocated() / 1024 ** 2  # 当前分配的显存
    reserved = torch.cuda.memory_reserved() / 1024 ** 2    # 当前保留的显存（包括缓存）
    max_allocated = torch.cuda.max_memory_allocated() / 1024 ** 2  # 历史最大分配
    max_reserved = torch.cuda.max_memory_reserved() / 1024 ** 2    # 历史最大保留
    print(f"[GPU 显存] 当前分配: {allocated:.2f} MB，当前保留: {reserved:.2f} MB，"
          f"最大分配: {max_allocated:.2f} MB，最大保留: {max_reserved:.2f} MB")





parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

infer_opts(parser)

parser.add_argument("--pooling", choices=["mean", "max", "first", "last"], default="first",
                    help="Pooling type.")

parser.add_argument("--labels_num", default=100, type=int, 
                    help="Number of prediction labels.")

parser.add_argument("--tokenizer", choices=["bert", "char", "space"], default="bert",
                    help="Specify the tokenizer." 
                         "Original Google BERT uses bert tokenizer on Chinese corpus."
                         "Char tokenizer segments sentences into characters."
                         "Space tokenizer segments sentences into words according to space."
                         )

parser.add_argument("--output_logits", action="store_true", help="Write logits to output file.")
parser.add_argument("--output_prob", action="store_true", help="Write probabilities to output file.")


args = parser.parse_args()

# Load the hyperparameters from the config file.
args = load_hyperparam(args)

args.load_model_path = "./models/finetuned_model.bin"
args.vocab_path = "./models/encryptd_vocab.txt"
args.test_path = "./datasets/cstnet-tls1.3/packet/nolabel_test_dataset.tsv"
args.labels_num = 100
args.embedding = "word_pos_seg" 
args.encoder = "transformer"
args.mask = "fully_visible"
# Build tokenizer.
args.tokenizer = str2tokenizer[args.tokenizer](args)

# Build classification model and load parameters.
args.soft_targets, args.soft_alpha = False, False
args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = Classifier(args)
model = load_model(model, args.load_model_path).to(args.device)
print_gpu_memory()

# model = VarCNN((5000,5000),100).to('cuda')
# model.load_state_dict(torch.load("/root/autodl-tmp/HAAD-GA/DLWF_pytorch/trained_model/length_1000/varcnn_1000.pth"))
# model = torch.load('/root/autodl-tmp/HAAD-GA/DLWF_pytorch/trained_model/Tor_DF.pkl')
X_train, y_train, X_valid, y_valid, X_test, y_test = load_rimmer_dataset(input_size=5000, num_classes=100) #(124991, 1000, 1),(124991, 100)


# pdb.set_trace()


# 启动搜索
haad = HAAD(
    model=model,
    original_trace=X_train,
    numant=10,
    max_inject=500,  # 最多注入 5 个虚包
    max_iter=20
)

best_combination, best_score = haad.run(label=y_train)
print("最优注入位置组合：", best_combination)