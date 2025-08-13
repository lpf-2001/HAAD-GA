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
from data import load_rimmer_dataset
from data import MyDataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
max_features = 1
num_classes = 200


def log_config(id):
    l = open("log_configs.out", "a")
    l.write("\nID{} {}\n".format(id, datetime.utcnow().strftime('%d.%m')))
    l.writelines(open(torconf, 'r').readlines())
    l.close()

def log(id, s, dnn=None):
    print("> {}".format(s))
    if dnn == "CNN":
        l = open("./trained_model/cnn.out", "a")
    elif dnn == "LSTM":
        l = open('./trained_model/lstm.out', "a")
    elif dnn =="SDAE":
        l = open('./trained_model/sdae.out',"a")
    elif dnn == "ENSEMBLE":
        # print("Ensemble")
        l = open('./trained_model/ensemble.out',"a")
    elif dnn == "DF":
        # print("Ensemble")
        l = open('./trained_model/df.out',"a")
    elif dnn == "VARCNN":
        l = open('./trained_model/varcnn.out',"a")
    elif dnn == "LLM":
        l = open('./trained_model/llm.out',"a")
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

    X_train, y_train, X_valid, y_valid, X_test, y_test=load_rimmer_dataset(input_size=5000,num_classes=num_classes,test_ratio=test_ratio,val_ratio=val_ratio)
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



def train_model(model,learn_param,model_train=True,train_loader=None,val_loader=None,test_loader=None):
    
    # 基础参数设置
    maxlen = learn_param.as_int('maxlen')
    epochs = learn_param.as_int('nb_epochs')
    batch_size = learn_param.as_int('batch_size')
    model_type = learn_param['model_type']
    
    # 优化器参数设置
    optimizer = learn_param['optimizer']
    
    #加载数据
    train_loader, val_loader, test_loader = data_process(learn_param)

    if optimizer == "rmsprop":
        optimizer = optim.RMSprop(model.parameters(),
                                lr = learn_param[optimizer].as_float('learning_rate'))
    elif optimizer == "adamax":
        optimizer = optim.Adamax(model.parameters(),
                                lr = learn_param[optimizer].as_float('learning_rate'))
    elif optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(),
                            lr = learn_param[optimizer].as_float('learning_rate'),
                            momentum = learn_param[optimizer].as_float('momentum'),
                            weight_decay = learn_param[optimizer].as_float('decay'))
    criterion = nn.CrossEntropyLoss()
    
    best_f1 = 0
    for epoch in range(epochs):
        model.train()
        count = 0
        accuracy = 0        
        with tqdm(train_loader, unit="batch") as tepoch:
            for batch_x, batch_y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                # print(f"batch_x shape:{batch_x.shape}") 
                tepoch.set_description(f"Epoch {epoch+1}")
                count = count + batch_y.size(0)
                optimizer.zero_grad()

                batch_x_tensor, batch_y_tensor = batch_x.float().to(device), batch_y.float().to(device)
                with autocast():
                    outputs = model(batch_x_tensor)
                    loss = criterion(outputs,batch_y_tensor.argmax(1))
                batch_accuracy= (outputs.argmax(1) == batch_y_tensor.argmax(1)).sum().item()/batch_y_tensor.shape[0]
                accuracy = accuracy + (outputs.argmax(1) == batch_y_tensor.argmax(1)).sum().item()
                loss.backward()
                optimizer.step()
                tepoch.set_postfix(loss=loss, accuracy=batch_accuracy)
            log(None,"epoch {}> loss:{}, accuracy:{}\n".format(epoch,loss, accuracy/count),model_type)

        if epoch%5==0:
            #验证模型
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for inputs, labels in tqdm(val_loader,desc="validation"):
                    # print(f"inputs shape:{inputs.shape}")
                    if torch.cuda.is_available():
                        inputs, labels = inputs.float().to(device), labels.float().to(device)
                        outputs = model(inputs)
                    loss = criterion(outputs, labels.argmax(1))
                    val_loss += loss.item()
                    val_total += labels.size(0)
                    val_correct += (outputs.argmax(1) == labels.argmax(1)).sum().item()
                val_accuracy = val_correct / val_total
                print(f"Validation Loss: {val_loss / len(val_loader)}, Validation Accuracy: {val_accuracy}%")
                log(None,f"Validation Loss: {val_loss / len(val_loader)}, Validation Accuracy: {val_accuracy}%\n",model_type)
            
            
            
            count = 0
            recall = 0
            f1 = 0
            precision = 0
            accuracy = 0
            with torch.no_grad():
                for inputs, labels in tqdm(test_loader):
                    inputs, labels = inputs.float(), labels.float()
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs).argmax(1).detach().cpu().numpy()       
                    labels = labels.argmax(1).detach().cpu().numpy()
                    count = count+1
                    # 1. 准确率 (Accuracy)
                    accuracy += accuracy_score(labels, outputs)
                    
                    # 2. 召回率 (Recall)
                    recall += recall_score(labels, outputs, average='weighted')
                    
                    # 3. 精确率 (Precision)
                    precision += precision_score(labels, outputs, average='weighted')

                    # 4. F1-score
                    f1 += f1_score(labels, outputs, average='weighted')
            log(None,"epoch {}> loss:{}, accuracy:{}\n".format(epoch,loss, accuracy/count),model_type) 
            log(None,f"Validation Loss: {val_loss / len(val_loader)}, Validation Accuracy: {val_accuracy}\n",model_type)
            log(None,f"Accuracy: {accuracy/count:.3f},Recall: {recall/count:.3f},Precision: {precision/count:.3f},F1-score: {f1/count:.3f}\n",model_type)
            if f1 > best_f1:
                if model_type == "CNN": 
                    torch.save(model.state_dict(),"./trained_model/length_5000/cnn_5000.pth")    
                elif model_type == "LSTM":
                    torch.save(model.state_dict(),"./trained_model/length_5000/lstm_5000.pth")
                elif model_type == "SDAE":
                    torch.save(model.state_dict(),"./trained_model/length_5000/sdae_5000.pth")
                elif model_type == "ENSEMBLE":
                    torch.save(model.state_dict(),"./trained_model/length_5000/ensemble_5000.pth")
                elif model_type == "DF":
                    torch.save(model.state_dict(),"./trained_model/length_5000/df_5000.pth") 
                elif model_type == "VARCNN":
                    torch.save(model.state_dict(),"./trained_model/length_5000/varcnn_5000.pth")
                elif model_type == "LLM":
                    torch.save(model.state_dict(),"./trained_model/length_5000/llm_5000.pth")
                best_f1 = f1
          

    





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
        model = MultiScaleLLM(input_len=5000, num_classes=num_classes,downsample_layers=3, use_mamba=False).to(device)
        train_model(model,config[model_type],model_train=model_train)
    



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train and test a deep neural network (SDAE, CNN or LSTM)')

    parser.add_argument('--model', '-m', default='cnn', type=str, choices=['cnn','lstm','sdae','ensemble','df','varcnn',"llm"],
                        help='choose model type cnn, lstm or sdae')
    # parser.add_argument('--check_input', '-cha', default=False, type=bool, choices=[True,False],
    #                     help='view dataset')
    parser.add_argument('--train_model', '-t', default=False, type=bool, choices=[True,False],
                        help='view dataset')

    args = parser.parse_args()

    if(args.train_model):
        main(args.model,args.train_model)

