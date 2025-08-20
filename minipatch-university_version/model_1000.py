import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm 
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Tor_cnn(nn.Module):
    def __init__(self):
        super(Tor_cnn,self).__init__()
        self.dropout = nn.Dropout(p=0.25)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32,kernel_size=2,padding='valid',stride=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64,kernel_size=2,padding='valid',stride=1)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128,kernel_size=4,padding='valid',stride=1)
        self.conv4 = nn.Conv1d(in_channels=128, out_channels=128,kernel_size=2,padding='valid',stride=1)
        self.maxpool1 = nn.MaxPool1d(kernel_size=2,padding=0)
        self.lstm = nn.LSTM(input_size=22, hidden_size=64, num_layers=1, batch_first=True)
        self.dense1 = nn.Linear(in_features=7808, out_features=256)
        self.dense2 = nn.Linear(in_features=256, out_features=128)
        self.dense3 = nn.Linear(in_features=128, out_features=100)
    def forward(self,x):
  
        x = x.transpose(1,2)
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = self.maxpool1(x)
        x = self.conv3(x)
        x = nn.functional.relu(x)
        x = self.maxpool1(x)
        x = self.conv4(x)
        x = nn.functional.relu(x)
        x = self.maxpool1(x)
        x = x.view(x.shape[0],-1)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        x = torch.sigmoid(x)
        x = nn.functional.softmax(x,dim=-1)
        return x
        
        

class Tor_lstm(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(Tor_lstm, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # 输入层 LSTM
        self.input_lstm = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)
        # 隐藏层的两层 LSTM
        self.hidden_lstm = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers, batch_first=True)
        # 输出层 LSTM
        self.output_lstm = nn.LSTM(hidden_size, hidden_size, num_layers=1, batch_first=True)
        # 全连接层
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # 输入层 LSTM
        out, _ = self.input_lstm(x)
        # 隐藏层的两层 LSTM
        out, _ = self.hidden_lstm(out)
        # 输出层 LSTM
        out, _ = self.output_lstm(out)
        # 全连接层
        out = self.fc(out[:, -1, :])  # 使用序列的最后一个时间步的输出作为全连接层的输入
        return out
    



# Define the autoencoder model
class Autoencoder(nn.Module):
    def __init__(self, in_dim, out_dim,enc_act, dec_act):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            getattr(nn, enc_act)()
        )
        self.decoder = nn.Sequential(
            nn.Linear(out_dim, in_dim),
            getattr(nn, dec_act)()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# Define the SAE model
class StackedAutoencoder(nn.Module):#输出(batch_size,100)
    def __init__(self, layers, nb_classes):
        super(StackedAutoencoder, self).__init__()
        self.encoders = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.num_layers = len(layers)
        # print(layers)
        for layer in layers:
            self.encoders.append(nn.Linear(in_features=layer.as_int('in_dim'), out_features=layer.as_int('out_dim')))
            if layer.as_float('dropout') > 0.0:
                self.dropouts.append(nn.Dropout(layer.as_float('dropout')))
            else:
                self.dropouts.append(None)
                
        self.classifier = nn.Linear(layers[-1].as_int('out_dim'), nb_classes)

    def forward(self, x):
        x = x.transpose(1,2)
        for i in range(self.num_layers):
            x = F.relu(self.encoders[i](x))
            if self.dropouts[i] is not None:
                x = self.dropouts[i](x)
        x = self.classifier(x).squeeze()
        # print("sdae shape:",x.shape)
        return x

# Autoencoder Training function
def train_model(i, train_loader,autoencoder,layer):
    # print(f"i:{i}")
    epochs = layer.as_int('epochs')
    optimizer_type = layer['optimizer']
    
    autoencoder.train()
    # Select the optimizer
    if optimizer_type == 'sgd':
        optimizer = optim.SGD(autoencoder.parameters(), lr=layer['sgd'].as_float('lr'), momentum=layer['sgd'].as_float('momentum'), weight_decay=layer['sgd'].as_float('decay'))
    elif optimizer_type == 'adam':
        optimizer = optim.Adam(autoencoder.parameters(), lr=layer['adam'].as_float('lr'), weight_decay=layer['adam'].as_float('decay'))
    elif optimizer_type == 'rmsprop':
        optimizer = optim.RMSprop(autoencoder.parameters(), lr=layer['rmsprop'].as_float('lr'), weight_decay=layer['rmsprop'].as_float('decay'))

    criterion = nn.MSELoss()
    autoencoder.train()
    for epoch in range(epochs):
        if i == 0:
            for data in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                # print(type(data))
                
                inputs, _ = data
                
                inputs = inputs.float().to(device).transpose(1,2)
                # print("sdae",inputs.shape)
                optimizer.zero_grad()
                # print(inputs.shape)
                outputs = autoencoder(inputs)
                loss = criterion(outputs, inputs)
                loss.backward()
                optimizer.step()
        else:
            for data in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                # print(type(data))
                # print(data)
                inputs = data[0]
                # print(inputs)
                # print(type(inputs))
                inputs = inputs.float().to(device)
                optimizer.zero_grad()
                # print(inputs.shape)
                outputs = autoencoder(inputs)
                loss = criterion(outputs, inputs)
                loss.backward()
                optimizer.step()
    return autoencoder

# Encoding data
def encode_data(i,data_loader,autoencoder):

    autoencoder.eval()
    encoded_data = []
    with torch.no_grad():
        if i==0:
            for data in data_loader:
                inputs, _ = data
                inputs = inputs.float().to(device).transpose(1,2)
                # print("encode_data",inputs.shape)
                encoded = autoencoder.encoder(inputs)
                # print(encoded.shape)
                encoded_data.append(encoded)
        else:
            for data in data_loader:
                inputs = data[0]
                inputs = inputs.float().to(device)
                # print("encode_data",inputs.shape)
                encoded = autoencoder.encoder(inputs)
                encoded_data.append(encoded)
    return torch.cat(encoded_data)

def make_layer(i, layer,  train_loader, test_loader, steps=0, gen=False):
    
    in_dim = layer.as_int('in_dim')
    out_dim = layer.as_int('out_dim')
    batch_size = layer.as_int('batch_size')
    enc_act = layer['enc_activation']
    dec_act = layer['dec_activation']

    autoencoder = Autoencoder(in_dim=in_dim, out_dim=out_dim,enc_act=enc_act,dec_act=dec_act).to(device)

    autoencoder = train_model(i, train_loader=train_loader,autoencoder=autoencoder,layer=layer)


    new_x_train1 = encode_data(i, train_loader, autoencoder=autoencoder)

    new_x_test1 = encode_data(i, test_loader, autoencoder=autoencoder)
    
    # 创建 TensorDataset
    dataset = TensorDataset(new_x_train1)
    train_loader = DataLoader(dataset, batch_size=steps, shuffle=True)

    dataset2 = TensorDataset(new_x_test1)
    test_loader = DataLoader(dataset2, batch_size=steps, shuffle=True)

    weights = autoencoder.encoder[0].weight.data

    return train_loader, test_loader, weights

def build_model(learn_params, train_gen, test_gen, steps=0, pre_train=True):
    layers = learn_params["layers"]
    nb_classes = learn_params.as_int('nb_classes')
    sae = StackedAutoencoder(layers,nb_classes=nb_classes).to(device)
    
    if pre_train:
        prev_x_train = train_gen
        prev_x_test = test_gen
        for i, layer in enumerate(layers):
            prev_x_train, prev_x_test, weights = make_layer(i,layer, prev_x_train, prev_x_test, steps=steps, gen=True)
            sae.encoders[i].weight.data = weights 
    
    return sae

class Tor_ensemble_model(nn.Module):
    def __init__(self,model1,model2,model3):
        super(Tor_ensemble_model, self).__init__()
        self.fc1 = nn.Linear(in_features=100, out_features=100, bias=False)
        self.fc2 = nn.Linear(in_features=100, out_features=100, bias=False)
        self.fc3 = nn.Linear(in_features=100, out_features=100, bias=False)
        self.model1 = model1
        self.model2 = model2
        self.model3 = model3
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):    
    
        x2 = self.model2(x)  #接受（batch_size,200,1） 
        x1 = self.model1(x)  
        x3 = self.model3(x)
        x1 = self.dropout(x1)
        x2 = self.dropout(x2)
        x3 = self.dropout(x3)
        # print(f"x1 shape:{x1.shape}")
    
        # 为每个张量应用不同的全连接层
        weighted_tensor1 = torch.sum(self.fc1(x1),0)/(x1.shape[0])
        # print(f"weight1 shape:{weighted_tensor1.shape}")
        weighted_tensor2 = torch.sum(self.fc1(x2),0)/(x1.shape[0])
        weighted_tensor3 = torch.sum(self.fc1(x3),0)/(x1.shape[0])
        weighted_tensor1 = weighted_tensor1.expand(x1.shape[0],x1.shape[1])
        weighted_tensor2 = self.fc2(x2).expand(x1.shape[0],x1.shape[1])
        weighted_tensor3 = self.fc3(x3).expand(x1.shape[0],x1.shape[1])
        # print(f"weight2 shape:{weighted_tensor1.shape}")




        # 对每个张量应用相应的权重进行逐元素相乘
        result1 = x1 * weighted_tensor1
        result2 = x2 * weighted_tensor2
        result3 = x3 * weighted_tensor3

        # 如果需要，可以将结果合并，例如逐元素求和
        final_result = result1 + result2 + result3
        return final_result




import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class DilatedBasicBlock1D(nn.Module):
    def __init__(self, filters1,filters2, layer, block, dilations):
        """
        A one-dimensional basic residual block with dilations.
        
        :param filters: the output's feature space
        :param layer: int representing the layer of this block (starting from 2)
        :param block: int representing this block (starting from 1)
        :param dilations: tuple representing amount to dilate first and second conv
        """
        super(DilatedBasicBlock1D, self).__init__()
        
        if layer == 2 or block != 1:
            stride = 1
        else:
            stride = 2
        
        self.conv1 = nn.Conv1d(in_channels=filters1, out_channels=filters2, kernel_size=3, 
                               stride=stride, padding=dilations[0], dilation=dilations[0], 
                               bias=False)
        self.bn1 = nn.BatchNorm1d(filters2)
        
        self.conv2 = nn.Conv1d(in_channels=filters2, out_channels=filters2, kernel_size=3, 
                               padding=dilations[1], dilation=dilations[1], 
                               bias=False)
        self.bn2 = nn.BatchNorm1d(filters2)

        # If layer > 2 and block == 1, perform downsample (shortcut)
        if layer > 2 and block == 1:
            self.downsample = nn.Sequential(
                nn.Conv1d(filters1, filters2, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(filters2)
            )
        else:
            self.downsample = None
    
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        
        # If downsample is applied
        if self.downsample is not None:
            residual = self.downsample(residual)
        
        out += residual
        out = F.relu(out)
        
        return out

class VarCNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(VarCNN, self).__init__()
        
        # First convolutional layer
        self.zero_padding = nn.ConstantPad1d(3,0)  # ZeroPadding1D(padding=3)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=7, stride=2, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        # Residual layers
        self.layer_blocks = [2, 2, 2, 2]
        self.features1 = 64
        self.features2 = 64
        self.blocks = self._make_layers()
        
        # Global average pooling
        self.avg_pool = nn.AdaptiveAvgPool1d(1)  # GlobalAveragePooling1D
        
        # Fully connected layer
        self.fc = nn.Linear(self.features2 // 2, num_classes)  # Dense layer

    def _make_layers(self):
        layers = []
        features1 = self.features1
        features2 = self.features2
        for i, blocks in enumerate(self.layer_blocks):

            layers.append(DilatedBasicBlock1D(features1,features2, i+2, 1, dilations=(1, 2)))
            for block in range(2, blocks+1):
                layers.append(DilatedBasicBlock1D(features2,features2, i+2, block, dilations=(4, 8)))
            features1 = features2
            features2 *= 2
        self.features2 = features2
     
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = x.transpose(1,2)
        x = self.zero_padding(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.blocks(x)
        x = self.avg_pool(x).squeeze(-1)
        x = self.fc(x)
        
        return x

class DFNet(nn.Module):
    def __init__(self, num_classes):
        super(DFNet, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=8, stride=1, padding=4),
            nn.BatchNorm1d(32),
            nn.ELU(),
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=8, stride=1, padding=4),
            nn.BatchNorm1d(32),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=8, stride=4, padding=2),
            nn.Dropout(0.1)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=8, stride=1, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=8, stride=1, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=8, stride=4, padding=2),
            nn.Dropout(0.1)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=8, stride=1, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=8, stride=1, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=8, stride=4, padding=3),
            nn.Dropout(0.1)
        )

        self.conv_block4 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=8, stride=1, padding=3),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=8, stride=1, padding=3),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=8, stride=4, padding=3),
            nn.Dropout(0.1)
        )

        self.fc1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.7)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.fc3 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = x.transpose(1,2)
        # print(x.shape)
        x = self.conv_block1(x)
        # print("block1:",x.shape)
        x = self.conv_block2(x)
        # print("block2:",x.shape)
        x = self.conv_block3(x)
        # print("block3:",x.shape)
        x = self.conv_block4(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x





