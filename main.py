import torch, time, os
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from models import *
from Losses import *
from Metrics import evaluate, cross_bound_check
from sklearn.preprocessing import StandardScaler
from algorithms import NESCQR, EnbPI
from utils import plot_PI, TimeSeriesDataLoader


args = {
    'window_size': 2,    # 时间序列数据的窗口长度
    'M': 3,              # 最终的集成模型的基学习器个数
    'max_epochs': 600,   # 模型最大遍历次数
    'l_rate': 1e-4,      # 学习率
    'batch_size': 1024,  # batch size
    'dropout': 0.2,      # 神经元丢弃率
    'replace': False,    # NESCQR的前向选择是否有放回
    'symmetric': True,   # conformity score是否对称
    'saveflag': True,    # 是否保存结果数据
    'save_dir': './results/',  # 结果保存路径
    'step': 2,           # DMCQR算法更新步长，int, 越小更新越快越准确
    'device': 'cuda',    # 使用的设备
    'verbose': True,     # 是否冗余输出
    'alpha_set': [0.05, 0.10, 0.15],  # 置信水平集合
    'activation_fn': 'tanh',          # 激活函数
    'num_repeat': 1,     # 每个类型的model有多少个
    'kernel_size': 2,    # TCN模型的卷积核大小
    'num_repeat': 1,     # 每个同类型的模型有多少个
}

def run_NESCQR(loader, df, X_train, Y_test, args, save_dir_NESCQR):
    """
    Run NESCQR model on the given data.
    
    Args:
        loader: Data loader object.
        df: Dataframe containing the data.
        X_train: Training data.
        Y_test: Testing data.
        args: Dictionary of arguments.
        save_dir_NESCQR: Directory to save NESCQR results.
    """

    # Define model
    alpha_base = max(args['alpha_set'])
    quantiles = [max(args['alpha_set'])/2, 1 - max(args['alpha_set'])/2]
    PINC = 100*(1 - np.array(args['alpha_set']))
    input_dim = X_train.shape[1]
    x_size = len(df.columns)
    out_dim = len(quantiles)
    hidden_units = [20 + i*4 for i in range(args['num_repeat'])]
    channel_sizes = [3 + i*2 for i in range(args['num_repeat'])]

    # NESCQR
    model_pool = [NET(input_dim, h, out_dim, args['activation_fn']) for h in hidden_units] + \
                [RNN(input_dim, h, out_dim, args['activation_fn'], args['device']) for h in hidden_units] + \
                [LSTM(input_dim, h, out_dim, args['device']) for h in hidden_units] + \
                [GRU(x_size, h, out_dim, args['device']) for h in hidden_units] + \
                [TCN(x_size, out_dim, [c]*2, args['kernel_size'], args['dropout']) for c in channel_sizes]
                
    label_pool = [f'BPNN_{h}' for h in hidden_units] + \
                [f'RNN_{h}' for h in hidden_units] + \
                [f'LSTM_{h}' for h in hidden_units] + \
                [f'GRU_{h}' for h in hidden_units] + \
                [f'TCN_{c}' for c in channel_sizes]

    nescqr = NESCQR(loader, model_pool, label_pool, args['batch_size'], args['M'], args['alpha_set'], 
                    args['l_rate'], args['max_epochs'], args['replace'], args['symmetric'], 
                    args['saveflag'], save_dir_NESCQR, alpha_base, args['step'], 
                    args['device'], args['verbose'])
    
    start_time = time.time()
    nescqr.fit()
    PI_nescqr = nescqr.predict()
    run_time = time.time() - start_time
    print(f'NESCQR run time: {run_time:.2f}s')

    print('Evaluating NESCQR...')
    Y_test_original = loader.inverse_transform(Y_test, is_label=True)
    res_nescqr = evaluate(Y_test_original, PI_nescqr, args['alpha_set'], saveflag=args['saveflag'], save_dir=save_dir_NESCQR)
    res_nescqr_cross = cross_bound_check(PI_nescqr, saveflag=args['saveflag'], save_dir=save_dir_NESCQR)

    print('Plotting prediction intervals constructed by NESCQR...')
    colors = 'darkorange'
    ind_show = range(0, 200)
    plot_PI(PI_nescqr, PINC, Y_test_original, 'DMCQRS', save_dir_NESCQR, args['saveflag'], ind_show, color=colors, \
            figsize=(16,12), fontsize=20,lw=0.5)
    print('NESCQR is done.')



def main():

    # 结果保存路径
    current_time    = time.strftime("%Y_%m_%d_%H_%M_%S")
    save_dir        = os.path.join("result", current_time)
    save_dir_NESCQR = os.path.join(save_dir, 'NESCQR')
    if not os.path.exists(save_dir_NESCQR):
        os.makedirs(save_dir_NESCQR)
    print(f'save_dir: {save_dir}')
        
    # Load data
    data_path = './data/Kaggle Wind Power Forecasting Data/Turbine_Data.csv'
    df        = pd.read_csv(data_path, parse_dates=["Unnamed: 0"])
    df        = df[['ActivePower', 'WindDirection','WindSpeed']]
    df.dropna(axis=0, inplace=True)
    df['WindDirection_sin'] = np.sin(df['WindDirection'])
    df['WindDirection_cos'] = np.cos(df['WindDirection'])
    df.drop('WindDirection', axis=1, inplace=True)
    # df = df.iloc[:1000]

    label_column = 'ActivePower'
    loader = TimeSeriesDataLoader(df, args['window_size'], label_column)
    X_train, Y_train = loader.get_train_data()
    X_val  , Y_val   = loader.get_val_data()
    X_test , Y_test  = loader.get_test_data()

    print(f'X_train.shape: {X_train.shape}, Y_train.shape: {Y_train.shape}')
    print(f'X_val.shape: {X_val.shape}, Y_val.shape: {Y_val.shape}')
    print(f'X_test.shape: {X_test.shape}, Y_test.shape: {Y_test.shape}')

    ## NESCQR
    run_NESCQR(loader,df, X_train, Y_test, args, save_dir_NESCQR)


if __name__ == '__main__':
    main()