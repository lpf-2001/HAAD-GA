import datetime
import pytz
import torch.nn as nn
import numpy as np
import os
import sys
import argparse
import torch
import torch.optim as optim
from torchsummary import summary
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset, random_split, Subset
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from tqdm import tqdm 
import sys 
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),'../utils'))
sys.path.append(parent_dir)

from configobj import ConfigObj
from model.model_5000 import *
from data import *

# torch.cuda.empty_cache()




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
max_features = 1
num_classes_dict = {
    "rimmer100": 100,
    "rimmer200": 200,
    "rimmer500": 500,
    "rimmer900": 900,
    "sirinam": 95
    }
num_classes = None
datatype = None



class SmoothCELoss(nn.Module):
    def __init__(self, s=0.001):
        super(SmoothCELoss, self).__init__()
        self.s = s

    def forward(self, logits, targets):
        """
        logits: [batch, C]，模型输出（未经过 softmax）
        targets: [batch]，类别标签（非 one-hot）
        """
        batch_size, num_classes = logits.size()

        # softmax 得到概率
        probs = F.softmax(logits, dim=1)

        # one-hot 编码
        y_onehot = F.one_hot(targets, num_classes=num_classes).float()

        # 第一项：标准交叉熵 (加权 (1-s))
        ce_loss = -(y_onehot * torch.log(probs + 1e-12)).sum(dim=1).mean()

        # 第二项：所有类别的 log(probs)，相当于熵惩罚 (加权 s)
        smooth_loss = -torch.log(probs + 1e-12).sum(dim=1).mean()

        # 最终 loss
        loss = (1 - self.s) * ce_loss + self.s * smooth_loss
        return loss


def log_config(id):
    l = open("log_configs.out", "a")
    l.write("\nID{} {}\n".format(id, datetime.utcnow().strftime('%d.%m')))
    l.writelines(open(torconf, 'r').readlines())
    l.close()

def log(id, s, dnn=None):
    print("> {}".format(s))
    if dnn == "CNN":
        l = open(f"./trained_model/cnn_{datatype}.out", "a")
    elif dnn == "LSTM":
        l = open(f'./trained_model/lstm_{datatype}.out', "a")
    elif dnn =="SDAE":
        l = open(f'./trained_model/sdae_{datatype}.out',"a")
    elif dnn == "ENSEMBLE":
        # print("Ensemble")
        l = open(f'./trained_model/ensemble_{datatype}.out',"a")
    elif dnn == "DF":
        # print("Ensemble")
        l = open(f'./trained_model/df_{datatype}.out',"a")
    elif dnn == "VARCNN":
        l = open(f'./trained_model/varcnn_{datatype}.out',"a")
    elif dnn == "LLM":
        l = open(f'./trained_model/llm_{datatype}.out',"a")
    if(id is not None):
        l.write("ID {} {}>\t{}\n".format(id,curtime().strftime('%H:%M:%S'),s))
    else:
        l.write(s)
    l.close()

def curtime():
    china_tz = pytz.timezone('Asia/Chongqing')
    return datetime.datetime.now(china_tz).time() #.%f')[:-3]

def gen_id():
    return datetime.date.today()

def data_process(learn_param):  

    # 基础参数设置
    batch_size = learn_param.as_int('batch_size')
    test_ratio = learn_param.as_float('test_ratio')
    val_ratio = learn_param.as_float('val_ratio')
    if datatype.startswith("rimmer"):    
        X_train, y_train, X_valid, y_valid, X_test, y_test=load_rimmer_dataset(input_size=5000,num_classes=num_classes,test_ratio=test_ratio,val_ratio=val_ratio)
    elif datatype.startswith("sirinam"):
        X_train, y_train, X_valid, y_valid, X_test, y_test=load_sirinam_dataset(input_size=5000,num_classes=num_classes,test_ratio=test_ratio,val_ratio=val_ratio)
    print("X_train shape:",X_train.shape)
    print("X_valid shape:",X_valid.shape)
    print("X_test shape:",X_test.shape)
    train_dataset = MyDataset(X_train,y_train)
    val_dataset = MyDataset(X_valid,y_valid)
    test_dataset = MyDataset(X_test,y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    
    return train_loader, val_loader, test_loader

          

@torch.inference_mode()
def evaluate(model, loader, criterion, device, calc_metrics=False):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for inputs, labels in tqdm(loader, desc="Evaluating", leave=False):
        inputs, labels = inputs.float().to(device), labels.float().to(device)
        targets = labels.argmax(1)  # 如果 labels 是 one-hot
        outputs = model(inputs)

        total_loss += criterion(outputs, targets).item() * labels.size(0)
        preds = outputs.argmax(1)

        correct += (preds == targets).sum().item()
        total += labels.size(0)

        if calc_metrics:
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())

    avg_loss = total_loss / total
    avg_acc = correct / total

    if calc_metrics:
        metrics_sum = {
            "accuracy": accuracy_score(all_labels, all_preds),
            "recall": recall_score(all_labels, all_preds, average='weighted'),
            "precision": precision_score(all_labels, all_preds, average='weighted'),
            "f1": f1_score(all_labels, all_preds, average='weighted')
        }
        return avg_loss, avg_acc, metrics_sum

    return avg_loss, avg_acc

def train_model(model, learn_param, model_train=True):
    # 参数
    epochs = learn_param.as_int('nb_epochs')
    model_type = learn_param['model_type']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据
    train_loader, val_loader, test_loader = data_process(learn_param)

    # 优化器
    opt_config = learn_param[learn_param['optimizer']]
    optimizers = {
        "rmsprop": optim.RMSprop,
        "adamax": optim.Adamax,
        "sgd": optim.SGD
    }
    opt_kwargs = {k: opt_config.as_float(k) for k in opt_config if k != 'optimizer'}
    optimizer = optimizers[learn_param['optimizer']](model.parameters(), **opt_kwargs)
    if model_type == "LLM":
        criterion = SmoothCELoss(s=0.005)
    else:
        criterion = nn.CrossEntropyLoss()
    best_f1 = 0
    save_paths = {
        "CNN": f"./trained_model/length_5000/cnn_{datatype}.pth",
        "LSTM": f"./trained_model/length_5000/lstm_{datatype}.pth",
        "SDAE": f"./trained_model/length_5000/sdae_{datatype}.pth",
        "ENSEMBLE": f"./trained_model/length_5000/ensemble_{datatype}.pth",
        "DF": f"./trained_model/length_5000/df_{datatype}.pth",
        "VARCNN": f"./trained_model/length_5000/varcnn_{datatype}.pth",
        "LLM": f"./trained_model/length_5000/llm_{datatype}.pth"
    }

    for epoch in range(epochs):
        model.train()
        total_correct, total_samples = 0, 0
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}") as tepoch:
            for batch_x, batch_y in tepoch:
                optimizer.zero_grad()
                batch_x, batch_y = batch_x.float().to(device), batch_y.float().to(device)
                with autocast():
                    outputs = model(batch_x)
                    
                    loss = criterion(outputs, batch_y.argmax(1))
                loss.backward()
                optimizer.step()

                preds = outputs.argmax(1)
                total_correct += (preds == batch_y.argmax(1)).sum().item()
                total_samples += batch_y.size(0)
                tepoch.set_postfix(loss=loss.item(), accuracy=total_correct/total_samples)

        log(None, f"Epoch {epoch+1} > loss: {loss.item()}, accuracy: {total_correct/total_samples:.4f}\n", model_type)

        # 验证 & 测试
        if (epoch+1) % 5 == 0:
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            log(None, f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}\n", model_type)

            _, _, metrics = evaluate(model, test_loader, criterion, device, calc_metrics=True)
            # 保存最优
            if metrics["f1"] > best_f1:
                torch.save(model.state_dict(), save_paths[model_type])
                best_f1 = metrics["f1"]
            log(None, f"Accuracy: {metrics['accuracy']:.3f}, Recall: {metrics['recall']:.3f}, "
                      f"Precision: {metrics['precision']:.3f}, F1-score: {metrics['f1']:.3f}\n", model_type)
            log(None, f"best_f1: {best_f1:.3f}\n", model_type)
            



def main(model_type,model_train):
    
    torconf = "My_tor_5000.conf"
    config = ConfigObj(torconf)
    
    if model_type == "cnn":
        model = Tor_cnn().to(device)
        train_model(model,config[model_type],model_train=model_train)
    elif model_type == "varcnn":
        model = VarCNN(5000,num_classes).to(device)
        train_model(model,config[model_type],model_train=model_train)
    elif model_type == "df":
        model = DFNet(num_classes).to(device)
        summary(model, input_size=(5000,1))
        # print(model)
        train_model(model,config[model_type],model_train=model_train)
    elif model_type == "lstm":
        model = Tor_lstm(input_size=config[model_type]['model_param'].as_int('input_size'),hidden_size=config[model_type]['model_param'].as_int('hidden_size'),num_layers=config[model_type]['model_param'].as_int('num_layers'),num_classes=num_classes).to(device)
        train_model(model,config[model_type],model_train=model_train)
    elif model_type == "sdae":
        learn_params = config[model_type]
        layers = [learn_params[str(x)] for x in range(1,learn_params.as_int('nb_layers')+1)]
        learn_params['layers'] = layers
        train_gen, val_gen, test_gen = data_process(learn_params)
        model = build_model(learn_params=learn_params,train_gen=train_gen,test_gen=val_gen, steps=learn_params.as_int('batch_size')
                            ).to(device)
        train_model(model,config[model_type],model_train=model_train,train_loader=train_gen,val_loader=val_gen,test_loader=test_gen)
    elif model_type == "ensemble":
        model1 = VarCNN(5000,num_classes).to(device)
        model2 = Tor_lstm(input_size=config["lstm"]['model_param'].as_int('input_size'),hidden_size=config["lstm"]['model_param'].as_int('hidden_size'),num_layers=config["lstm"]['model_param'].as_int('num_layers'),num_classes=config["lstm"]['model_param'].as_int('num_classes')).to(device)
        learn_params = config["sdae"]
        layers = [learn_params[str(x)] for x in range(1,learn_params.as_int('nb_layers')+1)]
        learn_params['layers'] = layers
        train_gen, val_gen, test_gen = data_process(learn_params)
        model3 = build_model(learn_params=learn_params,train_gen=train_gen,test_gen=val_gen, steps=learn_params.as_int('batch_size')
                            ).to(device)
        model = Tor_ensemble_model(model1,model2,model3).to(device)
        train_model(model,config[model_type],model_train=model_train)
    elif model_type == "llm":
        # model = LLM(input_len=5000, num_classes=100,embed_dim=64, conv_channels=128,downsample_layers=3, attn_heads=8).to(device)
        model = MultiScaleLLM_V2(num_classes=num_classes).to(device)
        train_model(model,config[model_type],model_train=model_train)
    



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train and test a deep neural network (SDAE, CNN or LSTM)')

    parser.add_argument('--model', '-m', default='cnn', type=str, choices=['cnn','lstm','sdae','ensemble','df','varcnn',"llm"],help='choose model type cnn, lstm or sdae')
    parser.add_argument('--dataset', '-d', default='rimmer100', type=str, choices=['rimmer100','rimmer200','rimmer500','rimmer900','sirinam'], help='view dataset')
    parser.add_argument('--train_model', '-t', default=False, type=bool, choices=[True,False],help='train the model or not')

    args = parser.parse_args()
    datatype = args.dataset
    num_classes = num_classes_dict[datatype]
    if(args.train_model):
        main(args.model,args.train_model)

