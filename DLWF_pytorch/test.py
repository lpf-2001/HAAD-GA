

# multiscale_llm_v2.py
import math
from typing import List, Optional
from tqdm import tqdm
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
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
# ----------------------------
# Helpers: sinusoidal pos enc
# ----------------------------


num_classes_dict = {
    "rimmer100": 100,
    "rimmer200": 200,
    "rimmer500": 500,
    "rimmer900": 900,
    "sirinam": 95
    }
num_classes = None
datatype = None



def sinusoidal_pos_encoding(L: int, D: int, device=None):
    """标准正弦位置编码，返回形状 [L, D]"""
    device = device if device is not None else torch.device("cpu")
    pe = torch.zeros(L, D, device=device)
    position = torch.arange(0, L, dtype=torch.float32, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, D, 2, dtype=torch.float32, device=device) * -(math.log(10000.0) / D))
    pe[:, 0::2] = torch.sin(position * div_term)
    if D % 2 == 1:
        # 奇数维最后一列用cos填充的一部分
        pe[:, 1::2] = torch.cos(position * div_term[:-1])
    else:
        pe[:, 1::2] = torch.cos(position * div_term)
    return pe  # [L, D]


# ----------------------------
# DropPath (stochastic depth)
# ----------------------------
class DropPath(nn.Module):
    """DropPath that drops entire residual paths (stochastic depth)."""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1 - self.drop_prob
        # shape: (batch, 1, 1) broadcastable
        shape = (x.shape[0],) + (1,) * (x.dim() - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        binary_mask = torch.floor(random_tensor)
        return x.div(keep_prob) * binary_mask


# ----------------------------
# DownBlock: conv residual with downsample
# ----------------------------
class DownBlock(nn.Module):
    """
    简单残差下采样模块：
      Conv1d -> BN -> GELU -> Conv1d(stride=2) -> BN, 残差（如果需要用 1x1 conv 投影）
    输出通道数不变，长度缩小约 2。
    """
    def __init__(self, channels: int, p_drop: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(channels, eps=1e-5)
        # 下采样
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(channels, eps=1e-5)
        self.act = nn.GELU()
        self.drop = nn.Dropout(p_drop)
        # 若输入长度为 L，输出为 floor((L+2*p - dilation*(k-1)-1)/stride +1)
        # 残差：需要把原序列下采样后再相加
        self.down_proj = nn.Conv1d(channels, channels, kernel_size=1, stride=2, bias=False)
        self.bn_proj = nn.BatchNorm1d(channels, eps=1e-5)

    def forward(self, x):
        # x: [B, C, L]
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act(out)
        out = self.drop(out)
        # proj identity
        id_proj = self.down_proj(identity)
        id_proj = self.bn_proj(id_proj)
        return out + id_proj  # [B, C, L//2]


# ----------------------------
# ScaleFusion (gated fusion across scales)
# ----------------------------
class ScaleFusion(nn.Module):
    """
    将不同尺度的特征加权融合成 [B, C, T]（假设传入的各尺度已经对齐到相同 T）
    输入: scales: list of tensors each [B, C, T]
    输出: fused [B, C, T]
    通过全局池化得到每尺度的权重（per-channel gating）
    """
    def __init__(self, channels: int, num_scales: int):
        super().__init__()
        self.num_scales = num_scales
        # 生成尺度权重的 MLP（共享）
        # 我们先用全局平均池化，得到 [B, C]，然后用小网络映射到 [B, num_scales]
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // 2 if channels // 2 > 0 else 1),
            nn.GELU(),
            nn.Linear(channels // 2 if channels // 2 > 0 else 1, num_scales),
        )

    def forward(self, scales: List[torch.Tensor]):
        # assume all have same [B, C, T]
        B, C, T = scales[0].shape
        stacked = torch.stack(scales, dim=1)  # [B, S, C, T]
        # compute per-scale descriptors: global pool over time
        avg = stacked.mean(dim=-1)  # [B, S, C]
        # use the channel-wise pooled descriptor for weighting: pool across channels
        # we will average channels to get per-scale scalar input
        avg_chan = avg.mean(dim=-1)  # [B, S]
        # map each scale's global descriptor to a score via mlp applied per scale using the C-dim pooled vector:
        # but mlp expects input [B, C], we instead use channel-avg descriptor as feature per scale
        # easier: for each scale, compute mlp on the channel-pooled vector along C:
        # We'll reshape to [B*S, 1] for a tiny mlp; to keep capacity, use avg over channels as feature
        features = avg_chan.view(B * self.num_scales, 1)  # [B*S, 1]
        # small linear mapping:
        # create a linear layer on-the-fly via parameterized mlp for single-dim input
        # For simplicity, reuse mlp but adapt input size by repeating
        # Map features -> logits for scales per batch element
        # Simpler approach: compute scale logits by: global_mean_per_scale -> softmax across scales
        scale_logits = avg_chan  # [B, S]
        weights = torch.softmax(scale_logits, dim=1)  # [B, S]
        weights = weights.view(B, self.num_scales, 1, 1)  # [B, S, 1, 1]
        fused = (stacked * weights).sum(dim=1)  # [B, C, T]
        return fused

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


# ----------------------------
# Main model: MultiScaleLLM_V2
# ----------------------------
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
        attn_layers: int = 4,          # 默认 4 层
        attn_dropout: float = 0.1,
        drop_path: float = 0.1,
        fuse_mode: str = 'gated'       # 'gated' | 'sum' | 'concat'
    ):
        super().__init__()
        assert fuse_mode in ('gated', 'sum', 'concat')
        self.fuse_mode = fuse_mode
        self.conv_channels = conv_channels
        self.downsample_layers = downsample_layers

        # ---- 输入是 [-1, +1] 的标量序列：用 Conv1d 当作 stem
        self.stem = nn.Sequential(
            nn.Conv1d(1, conv_channels, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(conv_channels, eps=1e-5),
            nn.GELU(),
            nn.Conv1d(conv_channels, conv_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(conv_channels, eps=1e-5),
            nn.GELU(),
        )

        # ---- 多尺度残差下采样
        self.down_blocks = nn.ModuleList([DownBlock(conv_channels, p_drop=0.05)
                                          for _ in range(downsample_layers)])

        # ---- 多尺度融合（learnable gated attention）
        if fuse_mode == 'concat':
            fused_channels = conv_channels * downsample_layers
        else:
            fused_channels = conv_channels

        self.scale_attn = None
        if fuse_mode == 'gated':
            self.scale_attn = ScaleFusion(conv_channels, downsample_layers)

        # projection to attention dim
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
                    nn.Linear(attn_dim, attn_dim * 4),
                    nn.GELU(),
                    nn.Dropout(attn_dropout),
                    nn.Linear(attn_dim * 4, attn_dim),
                ),
                "drop_path2": DropPath(dp_rates[i]),
                "drop": nn.Dropout(attn_dropout),
            }))

        # ---- 分类头
        self.head = nn.Sequential(
            nn.LayerNorm(attn_dim),
            nn.Linear(attn_dim, attn_dim * 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(attn_dim * 2, num_classes)
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
        else:  # 'gated'
            fused = self.scale_attn(aligned)  # [B, C, T]

        # 到序列维度 [B, T, C*? or C] 并线性投影到注意力维
        fused = fused.permute(0, 2, 1)  # [B, T, C*? or C]
        fused = self.proj(fused)  # [B, T, D]

        # 位置编码 + CLS
        B, T, D = fused.shape
        pe = self._pos_encoding(T, D, fused.device)  # [T, D]
        fused = fused + pe.unsqueeze(0)

        cls = self.cls_token.expand(B, -1, -1)  # [B, 1, D]
        seq = torch.cat([cls, fused], dim=1)  # [B, 1+T, D]

        # Transformer (Pre-Norm)
        for blk in self.blocks:
            x_norm = blk["norm1"](seq)
            attn_out, _ = blk["attn"](x_norm, x_norm, x_norm, need_weights=False)
            seq = seq + blk["drop_path1"](blk["drop"](attn_out))

            x_norm = blk["norm2"](seq)
            ffn_out = blk["ffn"](x_norm)
            seq = seq + blk["drop_path2"](blk["drop"](ffn_out))

        # 取 CLS 作为全局表示
        cls_out = seq[:, 0, :]  # [B, D]
        logits = self.head(cls_out)  # [B, num_classes]
        return logits

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
        l = open(f'./trained_model/llm2_{datatype}.out',"a")
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
        "LLM": f"./trained_model/length_5000/llm2_{datatype}.pth"
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
            


if __name__ == "__main__":
    torconf = "My_tor_5000.conf"
    config = ConfigObj(torconf)
    datatype = 'rimmer100'
    num_classes = num_classes_dict[datatype]
    model = MultiScaleLLM_V2(num_classes=100).to(device)
    train_model(model,config['llm'],model_train=True)