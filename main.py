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
    'window_size': 2,
    'M': 3,
    'max_epochs': 500,
    'l_rate': 1e-4,
    'batch_size': 512,
    'dropout': 0.2,
    'replace': False,
    'symmetric': True,
    'saveflag': True,
    'save_dir': './results/',
    'step': 2,
    'device': 'cuda',
    'verbose': True,
    'alpha_set': [0.05, 0.10, 0.15],
    'activation_fn': 'tanh',
    'num_repeat': 1,  # 每个类型的model有多少个
    'kernel_size': 2, # TCN模型的卷积核大小
    'num_repeat': 1,  # 每个同类型的模型有多少个
}


def main():

    # 结果保存路径
    current_time    = time.strftime("%Y_%m_%d_%H_%M_%S")
    save_dir        = os.path.join("result", current_time)
    save_dir_NESCQR = os.path.join(save_dir, 'NESCQR')
    if not os.path.exists(save_dir_NESCQR):
        os.makedirs(save_dir_NESCQR)
    print(f'save_dir: {save_dir_NESCQR}')
        
    # Load data
    data_path = './data/Kaggle Wind Power Forecasting Data/Turbine_Data.csv'
    df        = pd.read_csv(data_path, parse_dates=["Unnamed: 0"])
    df        = df[['ActivePower', 'WindDirection','WindSpeed']]
    df.dropna(axis=0, inplace=True)
    df['WindDirection_sin'] = np.sin(df['WindDirection'])
    df['WindDirection_cos'] = np.cos(df['WindDirection'])
    df.drop('WindDirection', axis=1, inplace=True)
    df = df.iloc[:1000]

    label_column = 'ActivePower'
    loader = TimeSeriesDataLoader(df, args['window_size'], label_column)
    X_train, Y_train = loader.get_train_data()
    X_val  , Y_val   = loader.get_val_data()
    X_test , Y_test  = loader.get_test_data()

    print(X_train.shape, Y_train.shape)
    print(X_val.shape, Y_val.shape)
    print(X_test.shape, Y_test.shape)

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
                
    # label_pool = ['BPNN']*num_repeat + ['RNN']*num_repeat + ['LSTM']*num_repeat + ['GRU']*num_repeat + ['TCN']*num_repeat
    label_pool = [f'BPNN_{h}' for h in hidden_units] + \
                [f'RNN_{h}' for h in hidden_units] + \
                [f'LSTM_{h}' for h in hidden_units] + \
                [f'GRU_{h}' for h in hidden_units] + \
                [f'TCN_{c}' for c in channel_sizes]

    nescqr = NESCQR(loader, model_pool, label_pool, args['batch_size'], args['M'], args['alpha_set'], 
                    args['l_rate'], args['max_epochs'], args['replace'], args['symmetric'], 
                    args['saveflag'], save_dir_NESCQR, alpha_base, args['step'], 
                    args['device'], args['verbose'])
    
    nescqr.fit()
    PI_nescqr = nescqr.predict()
    Y_test_original = loader.inverse_transform(Y_test, is_label=True)
    res_nescqr = evaluate(Y_test_original, PI_nescqr, args['alpha_set'], saveflag=args['saveflag'], save_dir=save_dir_NESCQR)
    res_nescqr_cross = cross_bound_check(PI_nescqr, saveflag=args['saveflag'], save_dir=save_dir_NESCQR)

    colors = 'darkorange'
    ind_show = range(0, 100)
    plot_PI(PI_nescqr, PINC, Y_test_original, 'DMCQRS', save_dir_NESCQR, args['saveflag'], ind_show, color=colors, \
            figsize=(16,12), fontsize=20,lw=0.5)
    print('NESCQR is done.')

if __name__ == '__main__':
    main()