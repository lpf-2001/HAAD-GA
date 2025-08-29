import argparse
import datetime
import numpy as np
import os
import pytz
import sys
import torch
import torch.nn as nn
import torch.optim as optim

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),'../utils'))
sys.path.append(parent_dir)


from data import *
from tqdm import tqdm 
from model.model import *
from configobj import ConfigObj
from torchsummary import summary
from torch.nn.utils import clip_grad_norm_
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset, random_split, Subset
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score




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
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim


class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        """
        pred: [B, num_classes]
        target: [B, num_classes] (one-hot or soft labels)
        """
        log_probs = F.log_softmax(pred, dim=-1)
        
        # 对 soft/one-hot target 做平滑
        # target_smooth = target * (1 - smoothing) + smoothing / num_classes
        num_classes = pred.size(1)
        target_smooth = target * (1.0 - self.smoothing) + self.smoothing / num_classes
        
        # cross-entropy
        loss = torch.mean(torch.sum(-target_smooth * log_probs, dim=-1))
        return loss


# ---- Mixup ----
def mixup_data(x, y, alpha=0.2):
    """
    x: [B, ...], y: [B]
    """
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ---- RandMask ----
def random_mask(x, mask_ratio=0.1):
    B, L = x.shape[:2]
    mask = torch.rand(B, L, device=x.device) > mask_ratio
    if x.ndim == 3:
        mask = mask.unsqueeze(-1)
    return x * mask.float()


# ---- Optimizer & Scheduler ----
def get_optimizer_scheduler(model, train_loader, num_epochs, lr=3e-4):
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    total_steps = len(train_loader) * num_epochs
    warmup_steps = int(0.05 * total_steps)

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + np.cos(np.pi * progress))  # Cosine decay

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return optimizer, scheduler


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
        l = open(f'./trained_model/new_llm_{datatype}.out',"a")
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
def evaluate(model, loader, device, calc_metrics=False):
    model.eval()
    correct, total = 0, 0
    all_preds, all_labels = [], []

    for inputs, labels in tqdm(loader, desc="Evaluating", leave=False):
        inputs, labels = inputs.float().to(device), labels.float().to(device)
        targets = labels.argmax(1)  # 如果 labels 是 one-hot
        outputs = model(inputs)
        preds = outputs.argmax(1)
        correct += (preds == targets).sum().item()
        total += labels.size(0)

        if calc_metrics:
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())

    avg_acc = correct / total

    if calc_metrics:
        metrics_sum = {
            "accuracy": accuracy_score(all_labels, all_preds),
            "recall": recall_score(all_labels, all_preds, average='weighted'),
            "precision": precision_score(all_labels, all_preds, average='weighted'),
            "f1": f1_score(all_labels, all_preds, average='weighted')
        }
        return avg_acc, metrics_sum

    return avg_acc

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
        "adamw": optim.AdamW,
        "sgd": optim.SGD
        
    }
    opt_kwargs = {k: opt_config.as_float(k) for k in opt_config if k != 'optimizer'}
    optimizer = optimizers[learn_param['optimizer']](model.parameters(), **opt_kwargs)
    if model_type == "LLM":
        criterion = LabelSmoothingLoss(smoothing=0.1)
        optimizer, scheduler = get_optimizer_scheduler(model, train_loader, epochs)
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
        "LLM": f"./trained_model/length_5000/new_llm_{datatype}.pth"
    }

    for epoch in range(epochs):
        model.train()
        total_correct, total_samples = 0, 0
        batch_, batch = 0, 0
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}") as tepoch:
            for batch_x, batch_y in tepoch:
                optimizer.zero_grad()
                batch_x, batch_y = batch_x.float().to(device), batch_y.float().to(device)
                batch_x = random_mask(batch_x, mask_ratio=0.1)
                batch_x, y_a, y_b, lam = mixup_data(batch_x, batch_y, alpha=0.2)
                with autocast():
                    logits = model(batch_x)
                    loss = mixup_criterion(criterion, logits, y_a, y_b, lam)
                loss.backward()
                clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                preds = logits.argmax(1)
                batch_ = (preds == batch_y.argmax(1)).sum().item()
                total_correct += batch_
                batch = batch_y.size(0)
                total_samples += batch
                tepoch.set_postfix(loss=loss.item(), accuracy=f"{100*batch_/batch:.2f}%")

        log(None, f"Epoch {epoch+1} > loss: {loss.item()}, accuracy: {100*total_correct/total_samples:.2f}%\n", model_type)

        # 验证 & 测试
        if (epoch+1) % 5 == 0:
            val_acc = evaluate(model, val_loader, device)
            log(None, f"Validation Accuracy: {100*val_acc:.2f}%\n", model_type)

            _, metrics = evaluate(model, test_loader, device, calc_metrics=True)
            # 保存最优
            if metrics["f1"] > best_f1:
                torch.save(model.state_dict(), save_paths[model_type])
                best_f1 = metrics["f1"]
            log(None, f"Accuracy: {100*metrics['accuracy']:.2f}%, Recall: {100*metrics['recall']:.2f}%, "
                      f"Precision: {100*metrics['precision']:.2f}%, F1-score: {100*metrics['f1']:.2f}%\n", model_type)
            log(None, f"best_f1: {100*best_f1:.2f}%\n", model_type)
            



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

