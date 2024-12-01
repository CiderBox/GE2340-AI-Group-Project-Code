import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
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

def train_with_cv(model, X, y, device, n_splits=10, epochs=300, batch_size=32, learning_rate=0.0005):
    """
    使用改进的早停机制和学习率调度进行训练
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    criterion = CombinedLoss().to(device)
    
    best_model = None
    best_val_loss = float('inf')
    
    # 早停参数（调整为更严格的值）
    initial_patience = 20  # 减少初始耐心值
    min_patience = 5      # 减少最小耐心值
    patience_decay = 0.7  # 加快耐心衰减
    min_lr = 1e-6
    improvement_threshold = 1e-4
    max_epochs_without_improvement = 50  # 添加最大无改善轮数限制
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        print(f"\nFold {fold + 1}/{n_splits}")
        
        # 数据准备
        X_train_fold = torch.FloatTensor(X[train_idx]).to(device)
        y_train_fold = torch.FloatTensor(y[train_idx]).to(device)
        X_val_fold = torch.FloatTensor(X[val_idx]).to(device)
        y_val_fold = torch.FloatTensor(y[val_idx]).to(device)
        
        # 重置模型
        if isinstance(model, (CNN, LSTM, MLP)):
            model.apply(weight_reset)
        
        # 初始化最佳模型为当前状态
        fold_best_model = copy.deepcopy(model.state_dict())
        fold_best_val_loss = float('inf')
        
        # 优化器设置
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        
        # 使用余弦退火重启调度器
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, 
            T_0=30,  # 减少重启周期
            T_mult=1,  # 保持固定重启周期
            eta_min=min_lr
        )
        
        # 早停相关变量
        patience = initial_patience
        patience_counter = 0
        no_improvement_streak = 0
        loss_history = []
        epochs_without_improvement = 0  # 添加无改善轮数计数
        min_val_loss = float('inf')
        
        # 计算初始验证损失
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_fold)
            if val_outputs.dim() == 1:
                val_outputs = val_outputs.unsqueeze(1)
            y_val_fold_reshaped = y_val_fold.view(-1, 1)
            initial_val_loss = criterion(val_outputs, y_val_fold_reshaped)
            fold_best_val_loss = initial_val_loss.item()
            min_val_loss = fold_best_val_loss
        
        # 训练循环
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            
            # 批次训练
            for i in range(0, len(X_train_fold), batch_size):
                batch_X = X_train_fold[i:i+batch_size]
                batch_y = y_train_fold[i:i+batch_size]
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                
                # 维度处理
                if outputs.dim() == 1:
                    outputs = outputs.unsqueeze(1)
                batch_y = batch_y.view(-1, 1)
                
                loss = criterion(outputs, batch_y)
                
                # L1正则化
                l1_lambda = 1e-5
                l1_norm = sum(p.abs().sum() for p in model.parameters())
                loss = loss + l1_lambda * l1_norm
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item()
            
            # 学习率调整
            current_lr = scheduler.get_last_lr()[0]
            scheduler.step()
            
            # 验证阶段
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_fold)
                if val_outputs.dim() == 1:
                    val_outputs = val_outputs.unsqueeze(1)
                y_val_fold = y_val_fold.view(-1, 1)
                val_loss = criterion(val_outputs, y_val_fold)
            
            val_loss_value = val_loss.item()
            loss_history.append(val_loss_value)
            
            # 检查是否有改善
            if val_loss_value < min_val_loss - improvement_threshold:
                min_val_loss = val_loss_value
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            
            # 计算相对改善
            relative_improvement = (fold_best_val_loss - val_loss_value) / fold_best_val_loss
            
            # 早停策略
            if relative_improvement > improvement_threshold:
                fold_best_val_loss = val_loss_value
                fold_best_model = copy.deepcopy(model.state_dict())
                patience_counter = 0
                no_improvement_streak = 0
                patience = min(initial_patience, patience * 1.1)
            else:
                patience_counter += 1
                no_improvement_streak += 1
                
                # 检查是否处于平台期
                if len(loss_history) > 5:
                    recent_losses = loss_history[-5:]
                    loss_std = np.std(recent_losses)
                    if loss_std < 1e-4 and current_lr > min_lr * 10:
                        print(f"Plateau detected with LR: {current_lr:.2e}")
                
                # 动态调整耐心值
                if no_improvement_streak > 5:  # 减少调整阈值
                    patience = max(min_patience, patience * patience_decay)
            
            # 早停检查
            if epochs_without_improvement >= max_epochs_without_improvement:
                print(f'Early stopping at epoch {epoch+1} due to no improvement for {max_epochs_without_improvement} epochs')
                break
            
            if patience_counter >= patience and current_lr <= min_lr * 2:
                print(f'Early stopping at epoch {epoch+1} due to patience exhausted')
                break
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(X_train_fold):.6f}, '
                      f'Val Loss: {val_loss_value:.6f}, LR: {current_lr:.2e}, '
                      f'Patience: {patience:.1f}, '
                      f'Epochs without improvement: {epochs_without_improvement}')
        
        # 确保fold_best_model不为None后再加载
        if fold_best_model is not None:
            model.load_state_dict(fold_best_model)
            # 更新全局最佳模型
            if fold_best_val_loss < best_val_loss:
                best_val_loss = fold_best_val_loss
                best_model = copy.deepcopy(fold_best_model)
    
    # 确保best_model不为None后再加载
    if best_model is not None:
        model.load_state_dict(best_model)
    
    return model

def evaluate_models(models, X_test, y_test, scaler, device):
    """
    评估所有模型的性能
    """
    results = {}
    
    # 准备测试数据
    X_test_2d = X_test.reshape(X_test.shape[0], -1)
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    
    for name, model in models.items():
        if isinstance(model, torch.nn.Module):
            model.eval()
            with torch.no_grad():
                predictions = model(X_test_tensor).cpu().numpy()
        else:  # SVR
            predictions = model.predict(X_test_2d)
        
        # 确保predictions是2D数组
        predictions = predictions.reshape(-1, 1)
        
        # 确保y_test是2D数组
        y_test_2d = y_test.reshape(-1, 1)
        
        # 为scaler.inverse_transform准备数据
        pred_padded = np.zeros((len(predictions), scaler.n_features_in_))
        pred_padded[:, 0] = predictions.ravel()
        
        y_test_padded = np.zeros((len(y_test_2d), scaler.n_features_in_))
        y_test_padded[:, 0] = y_test_2d.ravel()
        
        # 反向转换预测值和真实值
        predictions_orig = scaler.inverse_transform(pred_padded)[:, 0]
        y_test_orig = scaler.inverse_transform(y_test_padded)[:, 0]
        
        # 计算评估指标
        mse = mean_squared_error(y_test_orig, predictions_orig)
        mae = mean_absolute_error(y_test_orig, predictions_orig)
        r2 = r2_score(y_test_orig, predictions_orig)
        
        results[name] = {
            'predictions': predictions_orig,
            'mse': mse,
            'mae': mae,
            'r2': r2
        }
    
    return results

def plot_results(results, actual_values, dates, title='NASDAQ Index Prediction'):
    """
    可视化预测结果：生成两个独立图表，一个显示实际范围，一个从0开始显示
    """
    colors = ['red', 'blue', 'green', 'purple']
    
    # 第一个图：实际范围
    plt.figure(figsize=(15, 10))
    plt.plot(dates, actual_values, label='Actual', color='black', alpha=0.7, linewidth=2)
    for (name, result), color in zip(results.items(), colors):
        plt.plot(dates, result['predictions'], 
                label=f'{name}\nMSE: {result["mse"]:.2f}\nMAE: {result["mae"]:.2f}\nR²: {result["r2"]:.3f}', 
                color=color, alpha=0.6)
    
    plt.title(f'{title} (Actual Range)', fontsize=14, pad=20)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('NASDAQ Index', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('prediction_results_actual_range.png', bbox_inches='tight', dpi=300)
    plt.show()
    
    # 第二个图：从0开始的范围
    plt.figure(figsize=(15, 10))
    plt.plot(dates, actual_values, label='Actual', color='black', alpha=0.7, linewidth=2)
    for (name, result), color in zip(results.items(), colors):
        plt.plot(dates, result['predictions'], 
                label=f'{name}\nMSE: {result["mse"]:.2f}\nMAE: {result["mae"]:.2f}\nR²: {result["r2"]:.3f}', 
                color=color, alpha=0.6)
    
    plt.title(f'{title} (From Zero)', fontsize=14, pad=20)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('NASDAQ Index', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.xticks(rotation=45)
    plt.ylim(bottom=0)  # 设置y轴从0开始
    plt.tight_layout()
    plt.savefig('prediction_results_from_zero.png', bbox_inches='tight', dpi=300)
    plt.show()

def weight_reset(m):
    """
    重置模型权重的辅助函数
    """
    if isinstance(m, (nn.Conv1d, nn.Linear, nn.LSTM)):
        m.reset_parameters() 