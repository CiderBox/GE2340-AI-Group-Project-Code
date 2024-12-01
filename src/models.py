import torch
import torch.nn as nn
from sklearn.svm import SVR
import numpy as np

class CNN(nn.Module):
    def __init__(self, sequence_length, n_features):
        super(CNN, self).__init__()
        # 简化的CNN结构
        self.conv1 = nn.Conv1d(n_features, 64, kernel_size=3)
        self.conv2 = nn.Conv1d(64, 32, kernel_size=3)
        
        # 计算展平后的特征维度
        L1 = sequence_length - 2  # 第一层卷积后的长度
        L2 = L1 - 2  # 第二层卷积后的长度
        self.flatten_size = 32 * L2
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(self.flatten_size, 64)
        self.fc2 = nn.Linear(64, 1)
        
        # Batch Normalization
        self.batch_norm1 = nn.BatchNorm1d(64)
        self.batch_norm2 = nn.BatchNorm1d(32)
        
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        # 调整输入维度 (batch, features, sequence)
        x = x.permute(0, 2, 1)
        
        # 第一个卷积块
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        
        # 第二个卷积块
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = torch.relu(x)
        x = self.dropout(x)
        
        # 全连接层
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size=256, num_layers=3):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # 多层LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        # LSTM层
        lstm_out, _ = self.lstm(x)
        
        # 全连接层
        out = self.fc(lstm_out)
        return out

class MLP(nn.Module):
    def __init__(self, input_size):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        x = x.view(x.shape[0], -1)
        return self.layers(x) 