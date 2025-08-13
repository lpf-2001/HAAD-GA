import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm 
from torch.utils.data import DataLoader, TensorDataset

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



class LLM(nn.Module):
    """
    思路:
      1) Embedding-like map: 将 -1/1 -> 0/1 -> embed (via small conv)
      2) 多层 Conv1d 下采样（stride=2），把 seq_len 从 5000 降到 ~500
      3) 用 MultiheadAttention 在降采样后做全局交互（batch_first=True）
      4) Pooling + MLP 分类
    适配输入: x shape = (batch, seq_len, 1)  (值为 -1 / 1)
    输出: logits shape = (batch, num_classes)
    """
    def __init__(self, input_len=5000, num_classes=100,
                 embed_dim=64, conv_channels=128,
                 downsample_layers=3, attn_heads=8, attn_dropout=0.1):
        super().__init__()
        self.input_len = input_len

        # 0) 将 -1/1 -> 0/1 做索引，然后 embed 为 embed_dim
        # 用 Embedding 更方便（vocab=2）
        self.embedding = nn.Embedding(2, embed_dim)

        # 1) 初始 conv 将 embed_dim -> conv_channels
        # Conv1d expects (B, C, L)
        self.conv_in = nn.Conv1d(embed_dim, conv_channels, kernel_size=3, padding=1)

        # 2) 下采样卷积块（每层 stride=2）
        self.down_blocks = nn.ModuleList()
        cur_channels = conv_channels
        for i in range(downsample_layers):
            # conv -> bn -> relu, stride=2 下采样
            self.down_blocks.append(
                nn.Sequential(
                    nn.Conv1d(cur_channels, cur_channels, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm1d(cur_channels,eps=1e-5),
                    nn.ReLU(inplace=True),
                    # 1x conv to mix channels
                    nn.Conv1d(cur_channels, cur_channels, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm1d(cur_channels,eps=1e-5),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.01)
                )
            )
            # optionally increase channels at next stage
            # cur_channels *= 1  # keep same channels to save mem
        self.post_conv_proj = nn.Linear(cur_channels, cur_channels)  # small projection for attention

        # 3) Multi-head attention on downsampled sequence
        # MultiheadAttention in PyTorch expects (batch_first=True) available newer versions
        self.attn = nn.MultiheadAttention(embed_dim=cur_channels, num_heads=attn_heads,
                                          dropout=attn_dropout, batch_first=True)

        # 4) MLP head
        self.classifier = nn.Sequential(
            nn.LayerNorm(cur_channels),
            nn.Linear(cur_channels, cur_channels//2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(cur_channels//2, num_classes)
        )

    def forward(self, x):
        """
        x: (batch, seq_len, 1) with values -1 / 1
        """
        # squeeze last dim if present
        if x.ndim == 3 and x.size(-1) == 1:
            x = x.squeeze(-1)  # -> (batch, seq_len)

        # map -1/1 -> 0/1 indices for embedding
        x = ((x + 1) // 2).long()  # -1 -> 0, 1 -> 1, shape (B, L)

        emb = self.embedding(x)  # (B, L, embed_dim)
        # to conv format
        conv_in = emb.permute(0, 2, 1)  # (B, C_in, L)

        # initial conv
        out = self.conv_in(conv_in)  # (B, conv_channels, L)

        # downsample blocks
        for block in self.down_blocks:
            out = block(out)  # stride=2 halves length each time

        # out: (B, C, L_down)
        out = out.permute(0, 2, 1)  # (B, L_down, C)

        # optional small projection
        out = self.post_conv_proj(out)  # (B, L_down, C)

        # Attention (self-attn): query/key/value = out
        # MultiheadAttention with batch_first expects (B, L, E)
        attn_out, attn_weights = self.attn(out, out, out, need_weights=False)  # (B, L_down, C)

        # Pooling: 可以用 mean or attention pooling; 用 mean + LayerNorm
        pooled = attn_out.mean(dim=1)  # (B, C)

        logits = self.classifier(pooled)  # (B, num_classes)
        return logits
    
    
    


class MultiScaleLLM(nn.Module):
    """
    Multi-scale variant of your LLM:
      - keep embeddings for {-1, +1}
      - initial conv -> multiple downsample stages
      - collect multi-scale features (after each down block)
      - align spatial length (adaptive pooling) to the smallest scale and fuse
      - project fused multiscale feature to attn dim and apply attention or Mamba
      - pooling + MLP head

    参数要点：
      - scales: number of downsample stages to keep (>=1)
      - fuse_mode: 'concat' or 'sum' (默认 concat)
      - use_mamba: if True, will use MambaPlaceholder (replace with real Mamba)
    """

    def __init__(self,
                 input_len: int = 5000,
                 num_classes: int = 100,
                 embed_dim: int = 512,
                 conv_channels: int = 256,
                 downsample_layers: int = 3,
                 attn_heads: int = 8,
                 attn_dropout: float = 0.1,
                 fuse_mode: str = 'concat',
                 use_mamba: bool = False):
        super().__init__()
        assert fuse_mode in ('concat', 'sum')

        self.input_len = input_len
        self.embed = nn.Embedding(2, embed_dim)
        self.conv_in = nn.Conv1d(embed_dim, conv_channels, kernel_size=3, padding=1)

        # downsample blocks and we will keep the output feature of each block
        self.down_blocks = nn.ModuleList()
        cur_channels = conv_channels
        for i in range(downsample_layers):
            self.down_blocks.append(
                nn.Sequential(
                    nn.Conv1d(cur_channels, cur_channels, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm1d(cur_channels, eps=1e-5),
                    nn.ReLU(inplace=True),
                    nn.Conv1d(cur_channels, cur_channels, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm1d(cur_channels, eps=1e-5),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.01)
                )
            )
            # keep channels constant to save memory

        # After fusion we'll project to attn_dim
        self.fuse_mode = fuse_mode
        if fuse_mode == 'concat':
            fused_channels = cur_channels * downsample_layers
        else:
            fused_channels = cur_channels

        self.fuse_proj = nn.Linear(fused_channels, cur_channels)

        # Attention (or Mamba placeholder)
        self.attn = nn.MultiheadAttention(embed_dim=cur_channels, num_heads=attn_heads, dropout=attn_dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(cur_channels)
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(cur_channels, cur_channels*2),
            nn.GELU(),
            nn.Linear(cur_channels*2, cur_channels)
        )
        # 第二层归一化
        self.norm2 = nn.LayerNorm(cur_channels)
        # Dropout
        self.dropout = nn.Dropout(0.1)
        

        # classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(cur_channels),
            nn.Linear(cur_channels, max(cur_channels // 2, 1)),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(max(cur_channels // 2, 1), num_classes)
        )

    def forward(self, x: torch.Tensor):
        # x: (B, L, 1) with values -1 / 1
        if x.ndim == 3 and x.size(-1) == 1:
            x = x.squeeze(-1)  # (B, L)
        x = ((x + 1) // 2).long()
        emb = self.embed(x)  # (B, L, E)
        out = emb.permute(0, 2, 1)  # (B, C, L)

        out = self.conv_in(out)

        features: List[torch.Tensor] = []
        for block in self.down_blocks:
            out = block(out)  # (B, C, L_i)
            features.append(out)

        # features: list of tensors with shapes [(B,C,L1), (B,C,L2), ...]
        # align them to the smallest spatial length (last one)
        target_len = features[-1].size(-1)
        aligned = []
        for f in features:
            if f.size(-1) == target_len:
                aligned.append(f)
            else:
                # use adaptive avg pool to align length
                aligned.append(F.adaptive_avg_pool1d(f, output_size=target_len))

        # fuse along channel dim
        if self.fuse_mode == 'concat':
            fused = torch.cat(aligned, dim=1)  # (B, C*scales, L_target)
        else:
            # sum
            fused = torch.stack(aligned, dim=0).sum(dim=0)  # (B, C, L_target)

        # move to (B, L, C)
        fused = fused.permute(0, 2, 1)

        # project fused channels to attn channels
        fused = self.fuse_proj(fused)  # (B, L, C)
        # print("fused.shape",fused.shape)
 
        for i in range(0,3):
            attn_out, _ = self.attn(fused, fused, fused, need_weights=False)
            out1 = self.norm1(self.dropout(attn_out)+fused)
            ffn_out = self.ffn(out1)
            fused = self.norm2(self.dropout(ffn_out)+out1)
        
        # global pooling
        pooled = fused.mean(dim=1)
        logits = self.classifier(pooled)
        return logits
