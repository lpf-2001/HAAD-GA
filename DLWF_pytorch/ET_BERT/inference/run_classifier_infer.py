"""
  This script provides an exmaple to wrap UER-py for classification inference.
"""
import sys
import os
import argparse
import torch.nn as nn
import numpy as np
import torch

uer_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(uer_dir)

from uer.utils.constants import *
from uer.utils import *
from uer.utils.config import load_hyperparam
from uer.utils.seed import set_seed
from uer.model_loader import load_model
from uer.opts import infer_opts
from datasets.data import read_dataset
from fine_tuning.run_classifier import Classifier
from sklearn.model_selection import train_test_split

current_dir = os.path.dirname(os.path.abspath(__file__))
print("run_classifier_infer.py working directory:", current_dir)

def batch_loader(batch_size, src, seg, tgt):
    print(tgt.shape)
    instances_num = src.size()[0]
    for i in range(instances_num // batch_size):
        src_batch = src[i * batch_size : (i + 1) * batch_size, :]
        seg_batch = seg[i * batch_size : (i + 1) * batch_size, :]
        tgt_batch = tgt[i * batch_size : (i + 1) * batch_size]
        yield src_batch, seg_batch, tgt_batch
    if instances_num > instances_num // batch_size * batch_size:
        src_batch = src[instances_num // batch_size * batch_size :, :]
        seg_batch = seg[instances_num // batch_size * batch_size :, :]
        tgt_batch = tgt[instances_num // batch_size * batch_size :]
        yield src_batch, seg_batch, tgt_batch


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    infer_opts(parser)

    parser.add_argument("--pooling", choices=["mean", "max", "first", "last"], default="first",
                        help="Pooling type.")

    parser.add_argument("--labels_num", type=int, required=True,
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

    args.load_model_path = "models/finetuned_model.bin"
    args.vocab_path = "models/encryptd_vocab.txt"
    args.test_path = "datasets/cstnet-tls1.3/packet/nolabel_test_dataset.tsv"
    args.labels_num = 100
    args.embedding = "word_pos_seg" 
    args.encoder = "transformer"
    args.mask = "fully_visible"

    
    # Build tokenizer.
    args.tokenizer = str2tokenizer[args.tokenizer](args)

    # Build classification model and load parameters.
    args.soft_targets, args.soft_alpha = False, False
    model = Classifier(args)
    model = load_model(model, args.load_model_path)

    # For simplicity, we use DataParallel wrapper to use multiple GPUs.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        print("{} GPUs are available. Let's use them.".format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)

    src,tgt,seg,src_t,tgt_t,seg_t = read_dataset(args, "Rimmer")

    src_t = torch.LongTensor(src_t)
    tgt_t = np.array(tgt_t,dtype=np.int32)
    tgt_t = torch.LongTensor(tgt_t)
    seg_t = torch.LongTensor(seg_t)

    batch_size = args.batch_size
    instances_num = src_t.shape[0]

    print("The number of prediction instances: ", instances_num)

    model.eval()

    with open(args.prediction_path, mode="w", encoding="utf-8") as f:
        f.write("label")
        if args.output_logits:
            f.write("\t" + "logits")
        if args.output_prob:
            f.write("\t" + "prob")
        f.write("\n")
        correct = 0
        for i, (src_batch, seg_batch, tgt_batch) in enumerate(batch_loader(batch_size, src_t, seg_t, tgt_t)):
            src_batch = src_batch.to(device)
            seg_batch = seg_batch.to(device)
            tgt_batch = tgt_batch.to(device)
            with torch.no_grad():
                _, logits = model(src_batch, None, seg_batch)
            
            pred = torch.argmax(logits, dim=1)
            # print(pred[:10])
            correct = correct+(pred==tgt_batch).sum().item()
            # print(correct)
            prob = pred.cpu().numpy().tolist()
            true = tgt_batch.cpu().numpy().tolist()
            f.write("-------pred&true------")
            f.write('\n')
            f.write(str(prob))
            f.write('\n')
            f.write(str(true))
            f.write('\n')
            f.write(str(correct))
            f.write('\n')
            
        print(correct/instances_num)
            


if __name__ == "__main__":
    main()
