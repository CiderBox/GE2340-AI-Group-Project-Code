from data_loader import fetch_stock_data, prepare_data
from models import CNN, LSTM, MLP
from train_eval import train_model, evaluate_models, plot_results, train_with_cv
from sklearn.svm import SVR
import torch
import numpy as np

def main():
    # 检查是否有GPU可用，如果没有则使用CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 设置随机种子
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    
    # 获取2022-2024的数据
    data = fetch_stock_data(start_date='2022-01-01', end_date='2024-01-01')
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = prepare_data(data)
    
    # 合并训练集和验证集用于交叉验证
    X_train_full = np.concatenate([X_train, X_val], axis=0)
    y_train_full = np.concatenate([y_train, y_val], axis=0)
    
    # 获取特征数量
    n_features = X_train.shape[2]
    sequence_length = X_train.shape[1]
    
    # 初始化模型
    models = {
        'CNN': CNN(sequence_length, n_features).to(device),
        'LSTM': LSTM(input_size=n_features).to(device),
        'MLP': MLP(sequence_length * n_features).to(device),
        'SVR': SVR(kernel='rbf', C=100, epsilon=0.1, gamma='scale')
    }
    
    # 训练模型
    for name, model in models.items():
        print(f"\nTraining {name}...")
        if isinstance(model, torch.nn.Module):
            model = train_with_cv(
                model, 
                X_train_full, 
                y_train_full,
                device=device,  # 传入设备参数
                n_splits=3,
                epochs=100,
                batch_size=16,
                learning_rate=0.001
            )
        else:  # SVR
            model.fit(X_train_full.reshape(X_train_full.shape[0], -1), y_train_full.ravel())
    
    # 评估模型
    results = evaluate_models(models, X_test, y_test, scaler, device)
    
    # 获取测试集对应的日期
    test_dates = data.index[-(len(y_test)):]
    
    # 可视化结果
    actual_values = scaler.inverse_transform(
        np.concatenate([y_test.reshape(-1, 1), 
                       np.zeros((len(y_test), scaler.n_features_in_-1))], 
                      axis=1))[:, 0]
    
    plot_results(results, actual_values, test_dates)
    
    # 打印评估指标
    for name, result in results.items():
        print(f"\n{name} Results:")
        print(f"MSE: {result['mse']:.4f}")
        print(f"MAE: {result['mae']:.4f}")

if __name__ == "__main__":
    main() 