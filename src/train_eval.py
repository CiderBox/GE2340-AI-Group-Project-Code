import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import TimeSeriesSplit
import torch.optim as optim
import copy
from models import CNN, LSTM, MLP

class CombinedLoss(nn.Module):
    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()
        self.huber = nn.HuberLoss()
        
    def forward(self, pred, target):
        mse_loss = self.mse(pred, target)
        mae_loss = self.mae(pred, target)
        huber_loss = self.huber(pred, target)
        return 0.4 * mse_loss + 0.3 * mae_loss + 0.3 * huber_loss

def train_model(model, X_train, y_train, X_val, y_val, epochs=300, batch_size=64, learning_rate=0.0005):
    """
    优化后的训练函数
    """
    # 添加 device 检查
    device = next(model.parameters()).device
    
    # 确保数据在正确的设备上
    X_train = torch.FloatTensor(X_train).to(device)
    y_train = torch.FloatTensor(y_train).to(device)
    X_val = torch.FloatTensor(X_val).to(device)
    y_val = torch.FloatTensor(y_val).to(device)
    
    criterion = nn.MSELoss()
    
    # 使用AdamW优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, 
                                weight_decay=1e-4, betas=(0.9, 0.999))
    
    # 使用余弦退火学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=50, T_mult=2, eta_min=1e-6
    )
    
    # 转换为PyTorch张量
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train)
    X_val = torch.FloatTensor(X_val)
    y_val = torch.FloatTensor(y_val)
    
    # 创建数据加载器
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = TensorDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # 早停机制
    best_val_loss = float('inf')
    patience = 20
    patience_counter = 0
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        total_train_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y.view(-1, 1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        
        # 验证阶段
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = model(batch_X)
                val_loss = criterion(outputs, batch_y.view(-1, 1))
                total_val_loss += val_loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        scheduler.step()
        
        # 早停检查
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                model.load_state_dict(best_model_state)
                break
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}')

def evaluate_models(models, X_test, y_test, scaler, device):
    """
    评估所有模型，包括深度学习和传统机器学习模型
    """
    results = {}
    ensemble_predictions = []
    
    # 准备测试数据
    X_test_2d = X_test.reshape(X_test.shape[0], -1)
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    
    for name, model in models.items():
        if isinstance(model, torch.nn.Module):
            model.to(device)  # 确保模型在正确的设备上
            model.eval()
            with torch.no_grad():
                predictions = model(X_test_tensor).cpu().numpy()
        else:
            predictions = model.predict(X_test_2d)
            predictions = predictions.reshape(-1, 1)
        
        ensemble_predictions.append(predictions)
        
        # 只对收盘价进行反向转换
        predictions_orig = scaler.inverse_transform(
            np.concatenate([predictions, np.zeros((len(predictions), scaler.n_features_in_-1))], axis=1)
        )[:, 0]
        y_test_orig = scaler.inverse_transform(
            np.concatenate([y_test.reshape(-1, 1), np.zeros((len(y_test), scaler.n_features_in_-1))], axis=1)
        )[:, 0]
        
        mse = mean_squared_error(y_test_orig, predictions_orig)
        mae = mean_absolute_error(y_test_orig, predictions_orig)
        
        results[name] = {
            'predictions': predictions_orig,
            'mse': mse,
            'mae': mae
        }
    
    # 添加集成模型结果
    ensemble_pred = np.mean(ensemble_predictions, axis=0)
    ensemble_pred_orig = scaler.inverse_transform(
        np.concatenate([ensemble_pred, np.zeros((len(ensemble_pred), scaler.n_features_in_-1))], axis=1)
    )[:, 0]
    
    results['Ensemble'] = {
        'predictions': ensemble_pred_orig,
        'mse': mean_squared_error(y_test_orig, ensemble_pred_orig),
        'mae': mean_absolute_error(y_test_orig, ensemble_pred_orig)
    }
    
    return results

def plot_results(results, actual_values, dates, title='NASDAQ Index Prediction (2023-2024)'):
    """
    优化后的可视化函数，支持更多模型
    """
    plt.figure(figsize=(15, 10))
    plt.plot(dates, actual_values, label='Actual', color='black', alpha=0.7)
    
    # 为每个模型定义不同的颜色
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'cyan']
    
    for (name, result), color in zip(results.items(), colors):
        plt.plot(dates, result['predictions'], 
                label=f'{name}\nMSE: {result["mse"]:.2f}\nMAE: {result["mae"]:.2f}', 
                color=color, alpha=0.6)
    
    plt.title(title, fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('NASDAQ Index', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def train_with_cv(model, X, y, device, n_splits=5, epochs=150, batch_size=32, learning_rate=0.001):
    """
    使用时间序列交叉验证训练模型
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    criterion = CombinedLoss().to(device)
    
    best_model = None
    best_val_loss = float('inf')
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        print(f"\nFold {fold + 1}/{n_splits}")
        
        # 数据移至指定设备
        X_train_fold = torch.FloatTensor(X[train_idx]).to(device)
        y_train_fold = torch.FloatTensor(y[train_idx]).to(device)
        X_val_fold = torch.FloatTensor(X[val_idx]).to(device)
        y_val_fold = torch.FloatTensor(y[val_idx]).to(device)
        
        # 重置模型
        if isinstance(model, (CNN, LSTM, MLP)):
            model.apply(weight_reset)
        
        # 使用AdamW优化器和权重衰减
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        
        # 使用组合学习率调度器
        scheduler1 = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=learning_rate,
            epochs=epochs, steps_per_epoch=len(X_train_fold) // batch_size + 1
        )
        scheduler2 = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=10, factor=0.5
        )
        
        # 训练循环
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            
            for i in range(0, len(X_train_fold), batch_size):
                batch_X = X_train_fold[i:i+batch_size]
                batch_y = y_train_fold[i:i+batch_size]
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                
                # 计算损失
                loss = criterion(outputs, batch_y.view(-1, 1))
                
                # 添加L1正则化
                l1_lambda = 1e-5
                l1_norm = sum(p.abs().sum() for p in model.parameters())
                loss = loss + l1_lambda * l1_norm
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler1.step()
                
                total_loss += loss.item()
            
            # 验证
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_fold)
                val_loss = criterion(val_outputs, y_val_fold.view(-1, 1))
            
            scheduler2.step(val_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(X_train_fold):.6f}, Val Loss: {val_loss.item():.6f}')
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = copy.deepcopy(model.state_dict())
    
    model.load_state_dict(best_model)
    return model

def weight_reset(m):
    """
    重置模型权重
    """
    if isinstance(m, (nn.Conv1d, nn.Linear, nn.LSTM)):
        m.reset_parameters() 