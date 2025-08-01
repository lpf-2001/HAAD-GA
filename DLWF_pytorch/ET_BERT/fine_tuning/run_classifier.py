"""
This script provides an exmaple to wrap UER-py for classification.
"""

import argparse

import torch.nn as nn

import sys
import os
# 获取项目根目录（ET-BERT），并添加到sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from uer.layers import *
from uer.encoders import *
from uer.utils.vocab import Vocab
from uer.utils.constants import *
from uer.utils import *
from uer.utils.optimizers import *
from uer.utils.config import load_hyperparam
from uer.utils.seed import set_seed
from uer.model_saver import save_model
from uer.opts import finetune_opts
from datasets.data import read_dataset
from collections import Counter
import tqdm
import numpy as np
import pdb
from sklearn.model_selection import train_test_split

current_dir = os.path.dirname(os.path.abspath(__file__))
print("run_classification.py working directory:", current_dir)

import torch
import torch.nn as nn
import torch.nn.functional as F

class Classifier(nn.Module):
    def __init__(self, args):
        super().__init__()
        # 你自己的embedding和encoder初始化方法
        self.embedding = str2embedding[args.embedding](args, len(args.tokenizer.vocab))
        self.encoder = str2encoder[args.encoder](args)
        self.device = args.device
        self.labels_num = args.labels_num
        self.pooling = args.pooling

        hidden_size = args.hidden_size  # 768等
        target_length = args.seq_length              # 统一序列长度

        # 多尺度卷积 + 自适应池化
        self.conv = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(target_length)
        self.proj_conv = nn.Linear(2304, 768)
        self.proj_line2 = nn.Linear(768,128)
        self.output_layer_1 = nn.Linear(hidden_size, hidden_size)
        self.output_layer_2 = nn.Linear(hidden_size, self.labels_num)
        self.dropout = nn.Dropout(0.2)

    def process_emb(self, emb):
        # emb: [B, L, H] -> Conv1d expects [B, H, L]
        x = emb.transpose(1, 2)   # [B, H, L]
        x = self.conv(x)          # 卷积，长度L不变
        x = self.pool(x)          # 池化，长度变成 target_length
        x = x.transpose(1, 2)     # [B, target_length, H]
        return x


    def forward(self, src, tgt=None):
        if src.ndim>2:
            src = src.squeeze()
        src = src.long()
        src = (src[:,:5000]+1)/2
        # src: [B, L], seg: [B, L]
        emb1, emb2, emb3 = self.embedding(src)  # 多尺度输出，每个是 [B, L_i, H]
        # 卷积 + 池化统一长度128
        emb1 = self.process_emb(emb1)
        emb2 = self.process_emb(emb2)
        emb3 = self.process_emb(emb3)
        seg_resized = torch.ones((emb1.shape[0],emb1.shape[1])).to(self.device)
        # 编码
        out1 = self.encoder(emb1, seg_resized)
        out2 = self.encoder(emb2, seg_resized)
        out3 = self.encoder(emb3, seg_resized)
        concat = torch.cat([out1, out2, out3], dim=-1)  # [B, T, 2304]
        x = self.proj_conv(concat)      
        output = x + emb1  # 如果 emb1 是主要流，可以加残差

        # 池化
        if self.pooling == "mean":
            output = output.mean(dim=1)
        elif self.pooling == "max":
            output, _ = output.max(dim=1)
        elif self.pooling == "last":
            output = output[:, -1, :]
        else:
            output = output[:, 0, :]

        output = torch.sigmoid(self.output_layer_1(output))
        logits = self.output_layer_2(output)

        if tgt is None:
            return logits
        else:
            loss = nn.CrossEntropyLoss()(logits, tgt.view(-1))
            return loss, logits






def load_or_initialize_parameters(args, model):
    if args.pretrained_model_path is not None:
        # Initialize with pretrained model.
        model.load_state_dict(torch.load(args.pretrained_model_path, map_location={'cuda:1':'cuda:0', 'cuda:2':'cuda:0', 'cuda:3':'cuda:0'}), strict=False)
    else:
        # Initialize with normal distribution.
        for n, p in list(model.named_parameters()):
            if "gamma" not in n and "beta" not in n:
                p.data.normal_(0, 0.02)


def build_optimizer(args, model):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
    ]
    if args.optimizer in ["adamw"]:
        optimizer = str2optimizer[args.optimizer](optimizer_grouped_parameters, lr=args.learning_rate, correct_bias=False)
    else:
        optimizer = str2optimizer[args.optimizer](optimizer_grouped_parameters, lr=args.learning_rate,
                                                  scale_parameter=False, relative_step=False)
    if args.scheduler in ["constant"]:
        scheduler = str2scheduler[args.scheduler](optimizer)
    elif args.scheduler in ["constant_with_warmup"]:
        scheduler = str2scheduler[args.scheduler](optimizer, args.train_steps*args.warmup)
    else:
        scheduler = str2scheduler[args.scheduler](optimizer, args.train_steps*args.warmup, args.train_steps)
    return optimizer, scheduler


def batch_loader(batch_size, src, tgt):
    instances_num = src.size()[0]
    for i in range(instances_num // batch_size):
        src_batch = src[i * batch_size : (i + 1) * batch_size, :]
        tgt_batch = tgt[i * batch_size : (i + 1) * batch_size]
      

        yield src_batch, tgt_batch
    if instances_num > instances_num // batch_size * batch_size:
        src_batch = src[instances_num // batch_size * batch_size :, :]
        tgt_batch = tgt[instances_num // batch_size * batch_size :]

        yield src_batch, tgt_batch








def train_model(args, model, optimizer, scheduler, src_batch, tgt_batch):
    model.zero_grad()

    src_batch = src_batch.to(args.device)
    tgt_batch = tgt_batch.to(args.device)


    loss, _ = model(src_batch, tgt_batch)
    if torch.cuda.device_count() > 1:
        loss = torch.mean(loss)

    if args.fp16:
        with args.amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
    else:
        loss.backward()

    optimizer.step()
    scheduler.step()

    return loss


def evaluate(args, dataset, print_confusion_matrix=False):
    src = torch.LongTensor(dataset[0])
    tgt = np.array(dataset[1],dtype=np.int32)
    tgt = torch.LongTensor(tgt)


    batch_size = args.batch_size

    correct = 0
    # Confusion matrix.
    confusion = torch.zeros(args.labels_num, args.labels_num, dtype=torch.long)
    args.model.eval()
    for i, (src_batch, tgt_batch) in enumerate(batch_loader(batch_size, src, tgt)):
        src_batch = src_batch.to(args.device)
        tgt_batch = tgt_batch.to(args.device)

        with torch.no_grad():
            _, logits = args.model(src_batch, tgt_batch)
        pred = torch.argmax(nn.Softmax(dim=1)(logits), dim=1)
        gold = tgt_batch
        for j in range(pred.size()[0]):
            confusion[pred[j], gold[j]] += 1
        correct += torch.sum(pred == gold).item()


    if print_confusion_matrix:
        print("Confusion matrix:")
        print(confusion)
        cf_array = confusion.numpy()
        with open(current_dir+"/../results/confusion_matrix",'w') as f:
            for cf_a in cf_array:
                f.write(str(cf_a)+'\n')
        print("Report precision, recall, and f1:")
        eps = 1e-9
        for i in range(confusion.size()[0]):
            p = confusion[i, i].item() / (confusion[i, :].sum().item() + eps)
            r = confusion[i, i].item() / (confusion[:, i].sum().item() + eps)
            if (p + r) == 0:
                f1 = 0
            else:
                f1 = 2 * p * r / (p + r)
            print("Label {}: {:.3f}, {:.3f}, {:.3f}".format(i, p, r, f1))

    print("Acc. (Correct/Total): {:.4f} ({}/{}) ".format(correct / len(dataset[0]), correct, len(dataset[0])))
    return correct / len(dataset), confusion


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    finetune_opts(parser)

    parser.add_argument("--pooling", choices=["mean", "max", "first", "last"], default="first",
                        help="Pooling type.")

    parser.add_argument("--tokenizer", choices=["bert", "char", "space"], default="bert",
                        help="Specify the tokenizer."
                             "Original Google BERT uses bert tokenizer on Chinese corpus."
                             "Char tokenizer segments sentences into characters."
                             "Space tokenizer segments sentences into words according to space."
                             )

    parser.add_argument("--soft_targets", action='store_true',
                        help="Train model with logits.")
    parser.add_argument("--soft_alpha", type=float, default=0.5,
                        help="Weight of the soft targets loss.")
    
    args = parser.parse_args()

    # Load the hyperparameters from the config file.
    args = load_hyperparam(args)

    set_seed(args.seed)

    # Count the number of labels.
    # args.labels_num = count_labels_num(args.train_path)
    args.labels_num = 100
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Build tokenizer.
    args.tokenizer = str2tokenizer[args.tokenizer](args)

    # Build classification model.
    model = Classifier(args)

    # Load or initialize parameters.
    load_or_initialize_parameters(args, model)

    
    print(args.device)
    model = model.to(args.device)

    # Training phase.
    src,tgt,src_t,tgt_t = read_dataset(args, "Rimmer")
    batch_size = args.batch_size
    
    src = torch.LongTensor(src)
    tgt = np.array(tgt,dtype=np.int32)
    tgt = torch.LongTensor(tgt)

  
    instances_num = len(src)


    soft_tgt = None
    args.train_steps = int(instances_num * args.epochs_num / batch_size) + 1

    print("Batch size: ", batch_size)
    print("The number of training instances:", instances_num)

    optimizer, scheduler = build_optimizer(args, model)

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
        args.amp = amp

    if torch.cuda.device_count() > 1:
        print("{} GPUs are available. Let's use them.".format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)
    args.model = model

    total_loss, result, best_result = 0.0, 0.0, 0.0

    print("Start training.")

    for epoch in tqdm.tqdm(range(1, args.epochs_num + 1)):
        model.train()
        for i, (src_batch, tgt_batch) in enumerate(batch_loader(batch_size, src, tgt)):
           
            loss = train_model(args, model, optimizer, scheduler, src_batch, tgt_batch)
            total_loss += loss.item()
            if (i + 1) % args.report_steps == 0:
                print("Epoch id: {}, Training steps: {}, Avg loss: {:.3f}".format(epoch, i + 1, total_loss / args.report_steps))
                total_loss = 0.0

        result = evaluate(args,(src_t,tgt_t) )
        if result[0] > best_result:
            best_result = result[0]
            save_model(model, args.output_model_path)



if __name__ == "__main__":
    main()
    os.system("shutdown -h now")
