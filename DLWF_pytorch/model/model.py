import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm 
import math
from torch.utils.data import DataLoader, TensorDataset
from typing import List


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Tor_cnn(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Tor_cnn,self).__init__()
        self.dropout = nn.Dropout(p=0.1)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32,kernel_size=2,padding='valid',stride=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64,kernel_size=2,padding='valid',stride=1)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128,kernel_size=4,padding='valid',stride=1)
        self.conv4 = nn.Conv1d(in_channels=128, out_channels=256,kernel_size=2,padding='valid',stride=1)
        self.conv5 = nn.Conv1d(in_channels=256, out_channels=128,kernel_size=2,padding='valid',stride=1)
        self.maxpool1 = nn.MaxPool1d(kernel_size=2,padding=0)
        self.lstm = nn.LSTM(input_size=22, hidden_size=64, num_layers=1, batch_first=True)
        self.dense1 = nn.Linear(in_features=19840, out_features=2560)
        self.dense2 = nn.Linear(in_features=2560, out_features=128)
        self.dense3 = nn.Linear(in_features=128, out_features=out_dim)
    def forward(self,x):
  
        x = x.transpose(1,2)
        # x = self.dropout(x)
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
        x = self.conv5(x)
        x = nn.functional.relu(x)
        x = self.maxpool1(x)
        # x, (hn, cn) = self.lstm(x)
        # x = x[:,-1,:]
        # print(x.shape)
        x = x.view(x.shape[0],-1)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        # x = torch.sigmoid(x)
        
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
        optimizer = optim.SGD(autoencoder.parameters(), lr=layer['sgd'].as_float('learning_rate'), momentum=layer['sgd'].as_float('momentum'), weight_decay=layer['sgd'].as_float('decay'))
    elif optimizer_type == 'adamax':
        optimizer = optim.Adamax(autoencoder.parameters(), lr=layer['adamax'].as_float('learning_rate'))
    elif optimizer_type == 'rmsprop':
        optimizer = optim.RMSprop(autoencoder.parameters(), lr=layer['rmsprop'].as_float('learning_rate'))

    criterion = nn.MSELoss()
    autoencoder.train()
    for epoch in range(epochs):
        if i == 0:
            for data in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                # print(type(data))
                
                inputs, _ = data
                
                inputs = inputs.float().to(device).transpose(1,2)
                # print(inputs.shape)
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
                # print(inputs.shape)
                encoded = autoencoder.encoder(inputs)
                # print(encoded.shape)
                encoded_data.append(encoded)
        else:
            for data in data_loader:
                inputs = data[0]
                inputs = inputs.float().to(device)
                # print(inputs.shape)
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
    nb_classes = 95
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
        self.fc1 = nn.Linear(in_features=95, out_features=95, bias=False)
        self.fc2 = nn.Linear(in_features=95, out_features=95, bias=False)
        self.fc3 = nn.Linear(in_features=95, out_features=95, bias=False)
        self.dropout = nn.Dropout(0.1)
        self.model1 = model1
        self.model2 = model2
        self.model3 = model3

    def calculate_weighted_tensor(self, fc_layer, input_tensor):
        """
        计算加权张量
        :param fc_layer: 全连接层
        :param input_tensor: 输入张量
        :return: 加权张量
        """
        # 检查输入张量的维度
        if input_tensor.dim() != 2:
            raise ValueError("Input tensor should be 2-dimensional.")
        print(f"input shape:{input_tensor.shape}")
        batch_size = input_tensor.shape[0]
        # 应用全连接层并求和
        weighted_sum = torch.sum(fc_layer(input_tensor), dim=0)
        # 计算平均值
        weighted_tensor = weighted_sum / batch_size
        # 扩展维度以匹配输入张量的形状
        weighted_tensor = weighted_tensor.unsqueeze(0).expand(batch_size, -1)
        return weighted_tensor
    def forward(self, x):
        # 通过不同模型得到输出
        x1 = self.model1(x)
        
        x2 = self.model2(x)
        
        x3 = self.model3(x)
        x1 = self.dropout(x1)
        x2 = self.dropout(x2)
        x3 = self.dropout(x3)
        # 为每个张量应用不同的全连接层
        weighted_tensor1 = self.calculate_weighted_tensor(self.fc1, x1)
        weighted_tensor2 = self.calculate_weighted_tensor(self.fc2, x2)
        weighted_tensor3 = self.calculate_weighted_tensor(self.fc3, x3)

        # 对每个张量应用相应的权重进行逐元素相乘
        result1 = x1 * weighted_tensor1
        result2 = x2 * weighted_tensor2
        result3 = x3 * weighted_tensor3

        # 将结果合并，逐元素求和
        final_result = result1 + result2 + result3
        return final_result

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
            nn.Linear(4864, 5120),
            nn.BatchNorm1d(5120),
            nn.ReLU(),
            nn.Dropout(0.7)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(5120, 512),
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

    


class ScaleFusion(nn.Module):
    def __init__(self, channels: int, num_scales: int):
        super().__init__()
        # 时间注意力：学习每个时间点的重要性
        self.temporal_attn = nn.Sequential(
            nn.Conv1d(channels, 1, kernel_size=1),   # [B,1,T]
            nn.Softmax(dim=-1)                       # 时间维度归一化
        )
        # 尺度注意力：学习每个尺度的重要性
        self.scale_attn = nn.Sequential(
            nn.Linear(channels, channels // 4),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 4, 1)              # 输出 [B,k,1]
        )

    def forward(self, aligned: List[torch.Tensor]):  
        # aligned: k * [B,C,T]
        B, C, T = aligned[0].shape
        k = len(aligned)

        # ---- 时间注意力 ----
        ctx = []
        for a in aligned:
            # a: [B,C,T]
            score_t = self.temporal_attn(a)           # [B,1,T]
            ctx_a = (a * score_t).sum(dim=-1)         # [B,C] 时间加权和
            ctx.append(ctx_a)
        ctx = torch.stack(ctx, dim=1)                 # [B,k,C]

        # ---- 尺度注意力 ----
        scores = self.scale_attn(ctx)                 # [B,k,1]
        weights = torch.softmax(scores.squeeze(-1), dim=1)  # [B,k]

        # ---- 融合 ----
        aligned_stack = torch.stack(aligned, dim=1)   # [B,k,C,T]
        weights = weights.unsqueeze(-1).unsqueeze(-1) # [B,k,1,1]
        fused = (aligned_stack * weights).sum(dim=1)  # [B,C,T]

        return fused


# --------- 小组件 ----------

class SqueezeExcite1d(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(4, channels // reduction)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, channels, kernel_size=1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return x * self.fc(x)

class DepthwiseSeparableConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=7, stride=1, padding=None, bias=False):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.dw = nn.Conv1d(in_ch, in_ch, kernel_size, stride=stride, padding=padding,
                            groups=in_ch, bias=bias)
        self.pw = nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=bias)
    def forward(self, x):
        return self.pw(self.dw(x))

class DropPath(nn.Module):
    """Stochastic Depth. 按样本随机丢弃残差分支。"""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob
    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep).div_(keep)
        return x * mask

def sinusoidal_pos_encoding(L: int, C: int, device):
    pe = torch.zeros(L, C, device=device)
    position = torch.arange(0, L, dtype=torch.float, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, C, 2, device=device).float() * (-math.log(10000.0) / C))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe  # [L, C]

# --------- 改进后的下采样块 ----------

class DownBlock(nn.Module):
    """
    stride=2 下采样；深度可分离卷积 + SE + 残差。通道保持不变以节省显存。
    """
    def __init__(self, channels: int, p_drop: float = 0.1):
        super().__init__()
        self.conv1 = DepthwiseSeparableConv1d(channels, channels, kernel_size=7, stride=2)
        self.bn1   = nn.BatchNorm1d(channels, eps=1e-5)
        self.act1  = nn.GELU()
        self.conv2 = DepthwiseSeparableConv1d(channels, channels, kernel_size=5, stride=1)
        self.bn2   = nn.BatchNorm1d(channels, eps=1e-5)
        self.act2  = nn.GELU()
        self.se    = SqueezeExcite1d(channels)
        self.drop  = nn.Dropout(p_drop)
        self.pool_res = nn.AvgPool1d(kernel_size=2, stride=2)  # 残差支路下采样
    def forward(self, x):  # x: [B, C, L]
        identity = self.pool_res(x)
        out = self.conv1(x)
        out = self.bn1(out); out = self.act1(out)
        out = self.conv2(out)
        out = self.bn2(out); out = self.act2(out)
        out = self.se(out)
        out = self.drop(out)
        return out + identity  # [B, C, L/2]

# --------- 主模型 ----------

class MultiScaleLLM_V2(nn.Module):
    """
    改进版：数值输入→Conv stem→多尺度残差下采样→尺度注意力融合→
    位置编码→(浅层)Transformer→CLS池化→MLP分类头
    """
    def __init__(
        self,
        num_classes: int = 100,
        conv_channels: int = 128,
        downsample_layers: int = 3,
        attn_dim: int = 128,           # 注意力维度 = 通道数，简化对齐
        attn_heads: int = 4,
        attn_layers: int = 8,          # 先从 2~4 层开始，稳了再加深
        attn_dropout: float = 0.1,
        drop_path: float = 0.1,
        fuse_mode: str = 'gated'       # 'gated' | 'sum' | 'concat'
    ):
        super().__init__()
        assert fuse_mode in ('gated', 'sum', 'concat')
        self.fuse_mode = fuse_mode
        

        # ---- 输入是 [-1, +1] 的标量序列：用 Conv1d 当作 stem（比 Embedding 更贴合连续信号）
        self.stem = nn.Sequential(
            nn.Conv1d(1, conv_channels, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(conv_channels, eps=1e-5),
            nn.GELU(),
            nn.Conv1d(conv_channels, conv_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(conv_channels, eps=1e-5),
            nn.GELU(),
        )

        # ---- 多尺度残差下采样
        self.down_blocks = nn.ModuleList([DownBlock(conv_channels, p_drop=0.1)
                                          for _ in range(downsample_layers)])

        # ---- 多尺度融合（learnable gated attention）
        if fuse_mode == 'concat':
            fused_channels = conv_channels * downsample_layers
        else:
            fused_channels = conv_channels

        self.scale_attn = None
        # 标准化各尺度的全局上下文，生成尺度权重
        
        if fuse_mode == 'gated':
            self.scale_attn = ScaleFusion(fused_channels, downsample_layers)
        self.proj = nn.Linear(fused_channels, attn_dim)

        # ---- Transformer（Pre-Norm + DropPath）
        self.cls_token = nn.Parameter(torch.zeros(1, 1, attn_dim))
        self.pos_cache_len = 0
        self.pos_cache = None

        self.blocks = nn.ModuleList()
        dp_rates = torch.linspace(0, drop_path, attn_layers).tolist()
        for i in range(attn_layers):
            self.blocks.append(nn.ModuleDict({
                "norm1": nn.LayerNorm(attn_dim),
                "attn": nn.MultiheadAttention(embed_dim=attn_dim, num_heads=attn_heads,
                                              dropout=attn_dropout, batch_first=True),
                "drop_path1": DropPath(dp_rates[i]),
                "norm2": nn.LayerNorm(attn_dim),
                "ffn": nn.Sequential(
                    nn.Linear(attn_dim, attn_dim*4),
                    nn.GELU(),
                    nn.Dropout(attn_dropout),
                    nn.Linear(attn_dim*4, attn_dim),
                ),
                "drop_path2": DropPath(dp_rates[i]),
                "drop": nn.Dropout(attn_dropout),
            }))

        # ---- 分类头（更简洁）
        self.head = nn.Sequential(
            nn.LayerNorm(attn_dim),
            nn.Linear(attn_dim, attn_dim*2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(attn_dim*2, num_classes)
        )

        # 参数初始化
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def _pos_encoding(self, L: int, C: int, device):
        # 简单缓存，避免每步重算
        if self.pos_cache is None or self.pos_cache_len < L:
            pe = sinusoidal_pos_encoding(L, C, device)
            self.pos_cache = pe  # [L, C]
            self.pos_cache_len = L
        return self.pos_cache[:L]

    def forward(self, x: torch.Tensor):
        """
        x: [B, L] 或 [B, L, 1]，值为 -1/+1 的数值序列
        """
        if x.ndim == 3 and x.size(-1) == 1:
            x = x.squeeze(-1)          # [B, L]
        x = x.float()
        x = x.unsqueeze(1)             # [B, 1, L]

        # stem + 多尺度提取
        feat = self.stem(x)            # [B, C, L]
        features: List[torch.Tensor] = []
        y = feat
        for blk in self.down_blocks:
            y = blk(y)                 # [B, C, L_i]
            features.append(y)

        # 对齐长度（到最小尺度）
        target_len = features[-1].size(-1)
        aligned = [f if f.size(-1) == target_len
                   else F.adaptive_avg_pool1d(f, output_size=target_len) for f in features]  # k*[B,C,T]

        # 多尺度融合
        if self.fuse_mode == 'concat':
            fused = torch.cat(aligned, dim=1)  # [B, C*k, T]
        elif self.fuse_mode == 'sum':
            fused = torch.stack(aligned, dim=0).sum(dim=0)  # [B, C, T]
        else:# 'gated'
            fused = self.scale_attn(aligned)  # [B, C, 1]

        # 到序列维度 [B, T, C] 并线性投影到注意力维
        fused = fused.permute(0, 2, 1)                          # [B, T, C*? or C]
        fused = self.proj(fused)                                # [B, T, D]

        # 位置编码 + CLS
        B, T, D = fused.shape
        pe = self._pos_encoding(T, D, fused.device)             # [T, D]
        fused = fused + pe.unsqueeze(0)

        cls = self.cls_token.expand(B, -1, -1)                  # [B, 1, D]
        seq = torch.cat([cls, fused], dim=1)                    # [B, 1+T, D]

        # Transformer (Pre-Norm)
        for blk in self.blocks:
            x_norm = blk["norm1"](seq)
            attn_out, _ = blk["attn"](x_norm, x_norm, x_norm, need_weights=False)
            seq = seq + blk["drop_path1"](blk["drop"](attn_out))

            x_norm = blk["norm2"](seq)
            ffn_out = blk["ffn"](x_norm)
            seq = seq + blk["drop_path2"](blk["drop"](ffn_out))

        # 取 CLS 作为全局表示
        cls_out = seq[:, 0, :]                                   # [B, D]
        logits = self.head(cls_out)                              # [B, num_classes]
        return logits