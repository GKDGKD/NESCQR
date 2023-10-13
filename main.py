import torch, time, os
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from models import *
# from Losses import *
from Metrics import evaluate, cross_bound_check
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from algorithms import NESCQR, EnbPI, EnCQR
from utils import plot_PI, TimeSeriesDataLoader
from log.logutli import Logger


args = {
    'scaler'       : 'minmax',                 # 标准化方式，'minmax', 'standard'
    'train_ratio'  : 0.7,                      # 训练集比例
    'val_ratio'    : 0.15,                     # 验证集比例
    'test_ratio'   : 0.15,                     # 测试集比例
    'window_size'  : 2,                        # 时间序列数据的窗口长度
    'n_ensembles'  : 3,                        # NESCQR最终的集成模型的基学习器个数
    'max_epochs'   : 300,                      # 模型最大遍历次数
    'l_rate'       : 1e-4,                     # 学习率
    'batch_size'   : 1024,                     # batch size
    'dropout'      : 0.2,                      # 神经元丢弃率
    'replace'      : False,                    # NESCQR的前向选择是否有放回
    'symmetric'    : False,                    # conformity score是否对称
    'saveflag'     : True,                     # 是否保存结果数据
    'step'         : 2,                        # DMCQR算法更新步长，int, 越小更新越快越准确
    'device'       : 'cuda',                   # 使用的设备
    'verbose'      : True,                     # 是否冗余输出中间训练过程
    'alpha_set'    : [0.05, 0.10, 0.15],       # 置信水平集合
    'activation_fn': 'tanh',                   # 激活函数
    'hidden'       : 24,                       # 隐藏层神经元个数
    'channel_size' : 10,                       # TCN模型的卷积核个数
    'num_repeat'   : 1,                        # 每个类型的model有多少个
    'kernel_size'  : 2,                        # TCN模型的卷积核大小
    'num_repeat'   : 1,                        # 每个同类型的模型有多少个
}

def run_NESCQR(loader, x_size, args, save_dir_NESCQR, logger):
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
    logger.logger.info(f'NESCQR starts.')
    X_train, Y_train = loader.get_train_data(to_tensor=True)
    X_val  , Y_val   = loader.get_val_data(to_tensor=True)
    X_test , Y_test  = loader.get_test_data(to_tensor=True)
    alpha_base    = max(args['alpha_set'])
    quantiles     = [max(args['alpha_set'])/2, 1 - max(args['alpha_set'])/2]
    PINC          = 100*(1 - np.array(args['alpha_set']))
    input_dim     = X_train.shape[1]
    out_dim       = len(quantiles)
    hidden_units  = [args['hidden'] + i*4 for i in range(args['num_repeat'])]
    channel_sizes = [args['channel_size'] + i*2 for i in range(args['num_repeat'])]

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

    nescqr = NESCQR(model_pool, label_pool, args['batch_size'], args['n_ensembles'], args['alpha_set'], 
                    args['l_rate'], args['max_epochs'], args['replace'], args['symmetric'], 
                    alpha_base, args['step'], args['device'], logger, args['verbose'])
    
    start_time = time.time()
    nescqr.fit(X_train, Y_train, X_val, Y_val)
    PI_nescqr = nescqr.predict(X_val, Y_val, X_test, Y_test)
    run_time  = time.time() - start_time
    logger.logger.info(f'NESCQR run time: {run_time:.2f}s')

    logger.logger.info('Evaluating NESCQR...')
    Y_test_original  = loader.inverse_transform(Y_test, is_label=True)
    PI_nescqr        = loader.inverse_transform(PI_nescqr, is_label=True)
    res_nescqr       = evaluate(Y_test_original, PI_nescqr, args['alpha_set'], saveflag=args['saveflag'], save_dir=save_dir_NESCQR, Logger=logger)
    res_nescqr_cross = cross_bound_check(PI_nescqr, saveflag=args['saveflag'], save_dir=save_dir_NESCQR, Logger=logger)

    cols = [str(round(alpha/2, 3)) for alpha in args['alpha_set']] + \
            [str(round(1-alpha/2, 3)) for alpha in reversed(args['alpha_set'])]
    if args['saveflag']:
        df = pd.DataFrame(PI_nescqr, columns=cols)
        df.to_csv(os.path.join(save_dir_NESCQR,'PI_NESCQR.csv'), index=False)
        logger.logger.info(f'Confidence intervals saved in {save_dir_NESCQR}\conf_PIs.csv')

    logger.logger.info('Plotting prediction intervals constructed by NESCQR...')
    colors = 'darkorange'
    ind_show = range(0, 200) if len(PI_nescqr) > 200 else range(len(PI_nescqr))
    plot_PI(PI_nescqr, PINC, Y_test_original, 'NESCQR', save_dir_NESCQR, args['saveflag'], ind_show, color=colors, \
            figsize=(16,12), fontsize=20,lw=0.5)
    logger.logger.info('NESCQR is done.')


def run_EnbPI(loader, x_size, args, save_dir_enbpi, logger):

    logger.logger.info(f'EnbPI starts.')
    X_train, Y_train = loader.get_train_data(to_tensor=True)
    X_val  , Y_val   = loader.get_val_data(to_tensor=True)
    X_test , Y_test  = loader.get_test_data(to_tensor=True)
    X_train_enpi = torch.cat((X_train, X_val), axis=0)
    Y_train_enpi = torch.cat((Y_train, Y_val), axis=0)
    logger.logger.info(f'X_train_enpi.shape: {X_train_enpi.shape}, Y_train_enpi.shape: {Y_train_enpi.shape}')

    # Define models
    input_dim     = X_train.shape[1]
    hidden_units  = [args['hidden'] + i*4 for i in range(args['num_repeat'])]
    channel_sizes = [args['channel_size'] + i*2 for i in range(args['num_repeat'])]
    PINC          = 100*(1 - np.array(args['alpha_set']))

    model_pool_enbpi = [NET(input_dim, h, 1, args['activation_fn']) for h in hidden_units] + \
                [RNN(input_dim, h, 1, args['activation_fn'], args['device']) for h in hidden_units] + \
                [LSTM(input_dim, h, 1, args['device']) for h in hidden_units] + \
                [GRU(x_size, h, 1, args['device']) for h in hidden_units] + \
                [TCN(x_size, 1, [c]*2, args['kernel_size'], args['dropout']) for c in channel_sizes]
    label_model_pool = [f'NET_{h}' for h in hidden_units] + \
                [f'RNN_{h}' for h in hidden_units] + \
                [f'LSTM_{h}' for h in hidden_units] + \
                [f'GRU_{h}' for h in hidden_units] + \
                [f'TCN_{c}' for c in channel_sizes]
    
    ## Homogeneous models
    # model_pool_enbpi = [TCN(x_size, 1, [args['channel_size']]*2, args['kernel_size'], args['dropout'])] * args['n_ensembles']
    # label_model_pool = [f'TCN_{args["channel_size"]}']*3 # 全用TCN表现太好了，预测区间都快重合成一条线了
    # model_pool_enbpi = [LSTM(input_dim, args['hidden'], 1, args['device'])] * args['n_ensembles']
    # label_model_pool = [f'LSTM_{args["hidden"]}']*3  # LSTM表现也太好了，预测区间约等于回归预测加减一个小数

    logger.logger.info(f'model_pool_enbpi: {label_model_pool}')
    enbpi = EnbPI(model_pool_enbpi, args['alpha_set'], args['l_rate'], args['max_epochs'],
                   args['batch_size'], args['device'], logger, args['verbose'])
    
    start_time = time.time()
    enbpi.fit(X_train_enpi, Y_train_enpi)
    conf_PI_enbpi = enbpi.predict_interval(X_train_enpi, Y_train_enpi, X_test, Y_test, args['batch_size'])
    run_time = time.time() - start_time
    logger.logger.info(f'EnbPI run time: {run_time:.2f}s')

    logger.logger.info('Evaluating EnbPI...')
    Y_test_original = loader.inverse_transform(Y_test, is_label=True)
    conf_PI_enbpi   = loader.inverse_transform(conf_PI_enbpi, is_label=True)
    res_enbpi       = evaluate(Y_test_original, conf_PI_enbpi, args['alpha_set'], saveflag=args['saveflag'], save_dir=save_dir_enbpi, Logger=logger)
    res_enbpi_cross = cross_bound_check(conf_PI_enbpi, saveflag=args['saveflag'], save_dir=save_dir_enbpi, Logger=logger)

    cols = [str(round(alpha/2, 3)) for alpha in args['alpha_set']] + \
            [str(round(1-alpha/2, 3)) for alpha in reversed(args['alpha_set'])]
    if args['saveflag']:
        df = pd.DataFrame(conf_PI_enbpi, columns=cols)
        df.to_csv(os.path.join(save_dir_enbpi,'PI_EnbPI.csv'), index=False)
        logger.logger.info(f'Confidence intervals saved in {save_dir_enbpi}\conf_PIs.csv')

    logger.logger.info('Plotting prediction intervals constructed by EnbPI...')
    colors = 'darkorange'
    ind_show = range(0, 200) if len(conf_PI_enbpi) > 200 else range(len(conf_PI_enbpi))
    plot_PI(conf_PI_enbpi, PINC, Y_test_original, 'EnbPI', save_dir_enbpi, args['saveflag'], ind_show, color=colors, \
            figsize=(16,12), fontsize=20,lw=0.5)
    logger.logger.info('EnbPI is done.')

def run_EnCQR(loader, x_size, args, save_dir_encqr, logger):

    logger.logger.info(f'EnCQR starts.')
    X_train, Y_train = loader.get_train_data(to_tensor=True)
    X_val  , Y_val   = loader.get_val_data(to_tensor=True)
    X_test , Y_test  = loader.get_test_data(to_tensor=True)

    # Define models
    input_dim     = X_train.shape[1]
    hidden_units  = [args['hidden'] + i*4 for i in range(args['num_repeat'])]
    channel_sizes = [args['channel_size'] + i*2 for i in range(args['num_repeat'])]
    PINC          = 100*(1 - np.array(args['alpha_set']))

    out_dim_encqr = len(args['alpha_set']) * 2
    # model_pool_encqr = [NET(input_dim, h, out_dim_encqr, args['activation_fn']) for h in hidden_units] + \
    #             [RNN(input_dim, h, out_dim_encqr, args['activation_fn'], args['device']) for h in hidden_units] + \
    #             [LSTM(input_dim, h, out_dim_encqr, args['device']) for h in hidden_units] + \
    #             [GRU(x_size, h, out_dim_encqr, args['device']) for h in hidden_units] + \
    #             [TCN(x_size, out_dim_encqr, [c]*2, args['kernel_size'], args['dropout']) for c in channel_sizes]
    # label_model_pool = [f'NET_{h}' for h in hidden_units] + \
    #             [f'RNN_{h}' for h in hidden_units] + \
    #             [f'LSTM_{h}' for h in hidden_units] + \
    #             [f'GRU_{h}' for h in hidden_units] + \
    #             [f'TCN_{c}' for c in channel_sizes]
    
    model_pool_encqr = [NET(input_dim, h, out_dim_encqr, args['activation_fn']) for h in hidden_units] + \
                [LSTM(input_dim, h, out_dim_encqr, args['device']) for h in hidden_units] + \
                [TCN(x_size, out_dim_encqr, [c]*2, args['kernel_size'], args['dropout']) for c in channel_sizes]
    label_model_pool = [f'NET_{h}' for h in hidden_units] + \
                [f'LSTM_{h}' for h in hidden_units] + \
                [f'TCN_{c}' for c in channel_sizes]

    # Homogenous models
    # model_pool_encqr = [TCN(x_size, out_dim_encqr, [args['channel_size']]*2, args['kernel_size'], args['dropout'])] * args['n_ensembles']
    # label_model_pool = [f'TCN_{args["channel_size"]}']*3
    # model_pool_encqr = [LSTM(input_dim, args['hidden'], out_dim_encqr, args['device'])] * args['n_ensembles']
    # label_model_pool = [f'LSTM_{args["hidden"]}']*3 

    logger.logger.info(f'model_pool_encqr: {label_model_pool}')
    B = len(model_pool_encqr)
    batch_len = int(np.floor(X_train.shape[0]/B))
    # to_del = time_steps_in//time_steps_out # make sure there are no overlapping windows across batches
    to_del = 0

    train_data = []
    for b in range(len(model_pool_encqr)):
        # logger.info(f'b: {b}, batch_len: {batch_len}, b*batch_len:{b*batch_len}, (b+1)*batch_len-to_del: {(b+1)*batch_len-to_del}')
        train_data.append([X_train[b*batch_len:(b+1)*batch_len-to_del], Y_train[b*batch_len:(b+1)*batch_len-to_del]])

    encqr = EnCQR(model_pool_encqr, args['alpha_set'], args['batch_size'], args['batch_size'], args['l_rate'],\
                   args['max_epochs'], args['device'], logger, args['verbose'])
    
    start_time = time.time()
    encqr.fit(train_data, X_val, Y_val)
    PI_encqr, conf_PI_encqr = encqr.predict(X_test, Y_test)
    run_time = time.time() - start_time
    logger.logger.info(f'EnCQR run time: {run_time:.2f}s')

    logger.logger.info('Evaluating EnCQR...')
    Y_test_original = loader.inverse_transform(Y_test, is_label=True)
    conf_PI_encqr   = loader.inverse_transform(conf_PI_encqr, is_label=True)
    res_encqr       = evaluate(Y_test_original, conf_PI_encqr , args['alpha_set'], saveflag=args['saveflag'], save_dir=save_dir_encqr, Logger=logger)
    res_encqr_cross = cross_bound_check(conf_PI_encqr , saveflag=args['saveflag'], save_dir=save_dir_encqr, Logger=logger)

    cols = [str(round(alpha/2, 3)) for alpha in args['alpha_set']] + \
            [str(round(1-alpha/2, 3)) for alpha in reversed(args['alpha_set'])]
    if args['saveflag']:
        df = pd.DataFrame(conf_PI_encqr, columns=cols)
        df.to_csv(os.path.join(save_dir_encqr,'PI_EnCQR.csv'), index=False)
        logger.logger.info(f'Confidence intervals saved in {save_dir_encqr}\conf_PIs.csv')

    logger.logger.info('Plotting prediction intervals constructed by EnCQR...')
    colors = 'darkorange'
    ind_show = range(0, 200) if len(conf_PI_encqr) > 200 else range(len(conf_PI_encqr))
    plot_PI(conf_PI_encqr, PINC, Y_test_original, 'EnCQR', save_dir_encqr, args['saveflag'], ind_show, color=colors, \
            figsize=(16,12), fontsize=20,lw=0.5)
    logger.logger.info('EnCQR is done.')

def main():

    # Logger
    current_time = time.strftime("%Y_%m_%d_%H_%M_%S")
    save_dir     = os.path.join("result", current_time)
    log_id       = 'main'
    log_name     = f'Run_{current_time}.log'
    log_level    = 'info'
    logger       = Logger(log_id, save_dir, log_name, log_level)
    logger.logger.info("LOCAL TIME: " + \
        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    
    # 结果保存路径
    save_dir_NESCQR = os.path.join(save_dir, 'NESCQR')
    save_dir_enbpi  = os.path.join(save_dir, 'EnbPI')
    save_dir_encqr  = os.path.join(save_dir, 'EnCQR')
    if not os.path.exists(save_dir_NESCQR):
        os.makedirs(save_dir_NESCQR)
    if not os.path.exists(save_dir_enbpi):
        os.makedirs(save_dir_enbpi)
    if not os.path.exists(save_dir_encqr):
        os.makedirs(save_dir_encqr)
    logger.logger.info(f'save_dir: {save_dir}')

    logger.logger.info('Parameters: ')
    for k, v in args.items():
        logger.logger.info(f'{k}: {v}')
        
    # Load data
    logger.logger.info('Loading data...')
    data_path = './data/Kaggle Wind Power Forecasting Data/Turbine_Data.csv'
    df        = pd.read_csv(data_path, parse_dates=["Unnamed: 0"])
    df        = df[['ActivePower', 'WindDirection','WindSpeed']]
    df.dropna(axis=0, inplace=True)
    df['WindDirection_sin'] = np.sin(df['WindDirection'])
    df['WindDirection_cos'] = np.cos(df['WindDirection'])
    df.drop('WindDirection', axis=1, inplace=True)
    x_size = len(df.columns)
    # df = df.iloc[:1000]

    label_column = 'ActivePower'
    loader = TimeSeriesDataLoader(df, args['window_size'], label_column, args['scaler'],
                                  args['train_ratio'], args['val_ratio'], args['test_ratio'])
    X_train, Y_train = loader.get_train_data()
    X_val  , Y_val   = loader.get_val_data()
    X_test , Y_test  = loader.get_test_data()

    logger.logger.info(f'X_train.shape: {X_train.shape}, Y_train.shape: {Y_train.shape}')
    logger.logger.info(f'X_val.shape: {X_val.shape}, Y_val.shape: {Y_val.shape}')
    logger.logger.info(f'X_test.shape: {X_test.shape}, Y_test.shape: {Y_test.shape}')

    ## Run
    run_NESCQR(loader, x_size, args, save_dir_NESCQR, logger)
    run_EnbPI(loader, x_size, args, save_dir_enbpi, logger)
    run_EnCQR(loader, x_size, args, save_dir_encqr, logger)

    total_time = time.time() - current_time
    logger.logger.info(f'Finished. Total time used: {total_time}s, {total_time/60*60}h.')

if __name__ == '__main__':
    main()