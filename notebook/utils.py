from tkinter import font
import torch, os
import torch.nn as nn
import numpy as np
from Losses import PinballLoss
from torch.utils.data  import DataLoader, TensorDataset
from Metrics import metrics, cross_bound_check, cross_loss
import pandas as pd
import matplotlib.pyplot as plt
from models import QuantileRegressionEstimator, RegressionEstimator, EnbPI, EnCQR
from conformal_methods import CQRS, CQRA, DCQRS, DCQRA, MCQRS, MCQRA, DMCQRS, DMCQRA
from sklearn.model_selection import train_test_split

class data_loader: 
    def __init__(self, data_name:str, feature_selected:list, train_ratio=0.7,time_step=2,start_num=0,end_num=6000,x_size=3,**kwargs):
        '''
        定义数据集。Define the dataset.
        
        input parameters:
        data_name: the path of dataset, must be str.
        feature_selected: the features selected to used, list. Note that the first element of this list must be the response variable Y.
        特征选择列表，留下来的用来预测的变量，注意列表里的第一个元素对应Y（Y的历史数据也作为变量）。
        train_ratio: the ratio of training set.
        time_step: the order of lag, k阶历史数据（作为变量）。
        start_num: the begining position of the data, default as 0.
        end_num: the ending position of the data, e.g. 6000. If you want to use all data, set end_num = len(data) - time_step.
        x_size: the input dim of the data, which is the number of features (len(feature_selected)).
        '''
        data = pd.read_csv(data_name)
        data = data.dropna()
        # data_np = np.array(data.iloc[start_num:(end_num+time_step), feature_selected])
        data_np = np.array(data[feature_selected][start_num:(end_num+time_step)])
        data_min = data_np.min(axis=0)
        data_max = data_np.max(axis=0)
        data_1 = (data_np - data_min) / (data_max - data_min)
        data_1[data_1==0] = 0.00001
        data_1[data_1==1] = 0.99999
        data_len = end_num - start_num
        train_len = round(data_len*train_ratio)
        X = np.zeros(shape=(data_len,x_size*time_step))
        
        for i in range(time_step):
            X[:,i] = data_1[i:(end_num-start_num+i),0]
        for i in range(1,x_size):
            for j in range(time_step):
                X[:,(i*time_step+j)] = data_1[j:(end_num-start_num+j),i]
     
        self.y_train = data_1[time_step:(train_len+time_step),0].reshape(-1, 1)
        self.y_test = data_1[(train_len+time_step):,0].reshape(-1,).reshape(-1, 1)
        self.y_train_yuan = data_np[time_step:(train_len+time_step),0].reshape(-1, 1)
        self.y_test_yuan = data_np[(train_len+time_step):,0].reshape(-1, 1)
        
        self.X_train = X[0:train_len,:]
        self.X_test = X[train_len:,:]
        self.data_max = data_max
        self.data_min = data_min
        
    def anti_mixmax(self,y):
        '''
        Anti min-max normalization.
        '''
        data_anti = y*(self.data_max[0]-self.data_min[0])+self.data_min[0]
        return data_anti

class data_loader2: 
    def __init__(self, data, feature_selected:list, train_ratio=0.7,
    time_step=2,start_num=5000,end_num=6000,x_size=3,**kwargs):

        data_np = np.array(data[feature_selected][start_num:(end_num+time_step)])
        data_min = data_np.min(axis=0)
        data_max = data_np.max(axis=0)
        data_1 = (data_np - data_min) / (data_max - data_min)
        data_1[data_1==0] = 0.00001
        data_1[data_1==1] = 0.99999
        data_len = end_num - start_num
        train_len = round(data_len*train_ratio)
        X = np.zeros(shape=(data_len,x_size*time_step))
        
        for i in range(time_step):
            X[:,i] = data_1[i:(end_num-start_num+i),0]
        for i in range(1, x_size):
            for j in range(time_step):
                X[:,(i*time_step+j)] = data_1[j:(end_num-start_num+j),i]
     
        self.y_train = data_1[time_step:(train_len+time_step),0].reshape(-1, 1)
        self.y_test = data_1[(train_len+time_step):,0].reshape(-1, 1)
        self.y_train_yuan = data_np[time_step:(train_len+time_step),0].reshape(-1, 1)
        self.y_test_yuan = data_np[(train_len+time_step):,0].reshape(-1, 1)
        
        self.X_train = X[0:train_len,:]
        self.X_test = X[train_len:,:]
        self.data_max = data_max
        self.data_min = data_min
        
    def anti_mixmax(self,y):
        data_anti = y*(self.data_max[0]-self.data_min[0])+self.data_min[0]
        return data_anti

def transform(C, alpha_set, loader):
    '''
    将区间上下界映射回原来的scale，并输出评估结果。
    Anti-minmax the PIs results and output the metrics.
    
    input:
    C_lower: the lower bounds of prediction intervals (PIs), from small to large, np.array.
    C_upper: the uppper bounds of PIs, from small to large, np.array.
    alpha_set: the set of the confidence levels, list or np.array.
    loader: data loader, should conclude an anti-minmax functions.
    
    output:
    res: the evaluation metrics of the PIs, np.array, [num_alpha, 9].
    res_cross: MUCW, MLCW, cross loss of the PIs, check whether PIs cross, np.array.
    C_lower: the lower bounds of PIs after anti-minmax, from small to large, np.array.
    C_upper: the upper bounds of PIs after anti-minmax, from small to large, np.array.
    
    '''
    
    C = np.array(C)
    res, res_cross = [], []
    PINC = 1 - alpha_set
    C = loader.anti_mixmax(C)
    y_test_yuan = loader.y_test_yuan

    for i, alpha in enumerate(alpha_set):
        print('\nPINC = %d'%(100*PINC[i]))
        C_low = C[:, i]
        C_up = C[:, -(i+1)]
        # C_mean = (C_low + C_up) / 2   #[m]
        res.append(metrics(y_test_yuan, C_low, C_up, alpha))
        
    res_cross.append(cross_bound_check(C, y_test_yuan))
        
    return np.array(res), np.array(res_cross), C

def train(loader, model, model_regression, n_ensemble:int, alpha_set, max_epochs:int, batch_size, device:str, l_rate:float, ind_base:int, step:int, num_iter:int, verbose:bool):
    '''
    训练所有模型和算法。
    Train all models and conformal methods.

    input:
    loader: data loader, such as dataloader above.
    model: quantile regression model.
    model_regression: regression model for EnbPI algorithm.
    alpha_set: the list of conofidence levels, e.g. [0.1, 0.02, 0.3].
    l_rate: learning rate.
    ind_base: the baseline confidence level for MCQR type algorithm. 0 corresponds to alpha_set[0].
    step: the step size of DCQR type algorithm, the smaller it is, the faster the conformity scores update. int.
    num_iter: the number of training.
    verbose: redundant output, whether to output the process, bool.
 
    '''

    # Load data and convert to tensor
    print('Loading data...')
    X_train = torch.from_numpy(loader.X_train).to(torch.float32)
    X_test = torch.from_numpy(loader.X_test).to(torch.float32)
    Y_train = torch.from_numpy(loader.y_train).to(torch.float32) 
    Y_test = torch.from_numpy(loader.y_test).to(torch.float32)
    print(f'Y_train.shape = {Y_train.shape}, Y_test.shape = {Y_test.shape}')

    # Split the training data into 2 disjoint subsets.
    cal_ratio = 2/8
    x_train, x_cal, y_train, y_cal = train_test_split(X_train, Y_train, test_size=cal_ratio, shuffle=False)
    y_test_yuan = loader.y_test_yuan

    print(f'x_train.shape = {x_train.shape}, y_train.shape = {y_train.shape}')
    print(f'x_cal.shape = {x_cal.shape}, y_cal.shape = {y_cal.shape}')
    print(f'X_test.shape = {X_test.shape}, Y_test.shape = {Y_test.shape}')
    print(f'y_test_yuan.shape = {y_test_yuan.shape}')
    print('Data prepared.')

    metric_QR, cross_QR, metric_CQRS, cross_CQRS = [], [], [], []
    metric_CQRA, cross_CQRA, metric_DCQRS, cross_DCQRS = [], [], [], []
    metric_DCQRA, cross_DCQRA, metric_MCQRS, cross_MCQRS = [], [], [], []
    metric_MCQRA, cross_MCQRA, metric_DMCQRS, cross_DMCQRS = [], [], [], []
    metric_DMCQRA, cross_DMCQRA, metric_EnbPI, cross_EnbPI, metric_EnCQR, cross_EnCQR = [], [], [], [], [], []
    C_all = []  #PIs
    NN_regression = [model_regression]*n_ensemble

    for iter in range(num_iter):
        print('\nIter = %d, Quantile regression: '%(iter+1))
        QR_estimator = QuantileRegressionEstimator(model, alpha_set, max_epochs, batch_size, device, l_rate, verbose)
        QR_estimator.fit(x_train, y_train, x_cal, y_cal)

        res_cal = QR_estimator.predict(x_cal)
        res_test_QR = QR_estimator.predict(X_test)

        res_cal_base = res_cal[:, [ind_base, -(ind_base+1)]]
        res_test_base = res_test_QR[:, [ind_base, -(ind_base+1)]]

        res_QR, res_cross_QR, C_QR = transform(res_test_QR, alpha_set, loader)
        metric_QR.append(res_QR)
        cross_QR.append(res_cross_QR)

        print('\nCQRS: ')
        C_CQRS = CQRS(res_test_QR, res_cal, y_cal, alpha_set)
        res_CQRS, res_cross_CQRS, C_CQRS = transform(C_CQRS, alpha_set, loader)
        metric_CQRS.append(res_CQRS)
        cross_CQRS.append(res_cross_CQRS)

        print('\nCQRA: ')
        C_CQRA = CQRA(res_test_QR, res_cal, y_cal, alpha_set)
        res_CQRA, res_cross_CQRA, C_CQRA = transform(C_CQRA, alpha_set, loader)
        metric_CQRA.append(res_CQRA)
        cross_CQRA.append(res_cross_CQRA)

        print('\nDCQRS: ')
        C_DCQRS = DCQRS(res_test_QR, Y_test, res_cal, y_cal, alpha_set, step)
        res_DCQRS, res_cross_DCQRS, C_DCQRS = transform(C_DCQRS, alpha_set, loader)
        metric_DCQRS.append(res_DCQRS)
        cross_DCQRS.append(res_cross_DCQRS)

        print('\nDCQRA: ')
        C_DCQRA = DCQRA(res_test_QR, Y_test, res_cal, y_cal, alpha_set, step)
        res_DCQRA, res_cross_DCQRA, C_DCQRA = transform(C_DCQRA, alpha_set, loader)
        metric_DCQRA.append(res_DCQRA)
        cross_DCQRA.append(res_cross_DCQRA)

        print('\nMCQRS: ')
        C_MCQRS = MCQRS(res_cal_base, y_cal, res_test_base, alpha_set)
        res_MCQRS, res_cross_MCQRS, C_MCQRS = transform(C_MCQRS, alpha_set, loader)
        metric_MCQRS.append(res_MCQRS)
        cross_MCQRS.append(res_cross_MCQRS)

        print('\nMCQRA: ')
        C_MCQRA = MCQRA(res_cal_base, y_cal, res_test_base, alpha_set)
        res_MCQRA, res_cross_MCQRA, C_MCQRA = transform(C_MCQRA, alpha_set, loader)
        metric_MCQRA.append(res_MCQRA)
        cross_MCQRA.append(res_cross_MCQRA)

        print('\nDMCQRS: ')
        C_DMCQRS = DMCQRS(res_cal_base, y_cal, res_test_base, Y_test, alpha_set, step)
        res_DMCQRS, res_cross_DMCQRS, C_DMCQRS = transform(C_DMCQRS, alpha_set, loader)
        metric_DMCQRS.append(res_DMCQRS)
        cross_DMCQRS.append(res_cross_DMCQRS)

        print('\nDMCQRA: ')
        C_DMCQRA = DMCQRA(res_cal_base, y_cal, res_test_base, Y_test, alpha_set, step)
        res_DMCQRA, res_cross_DMCQRA, C_DMCQRA = transform(C_DMCQRA, alpha_set, loader)
        metric_DMCQRA.append(res_DMCQRA)
        cross_DMCQRA.append(res_cross_DMCQRA)

        print('\nEnbPI: ')
        enbpi = EnbPI(NN_regression, alpha_set, l_rate, max_epochs, batch_size=256, device=device, verbose=verbose)
        enbpi.fit(X_train, Y_train)
        C_enbpi = enbpi.conformal(X_train, Y_train, X_test, Y_test)
        res_enbpi, res_cross_enbpi, C_enbpi = transform(C_enbpi, alpha_set, loader)
        metric_EnbPI.append(res_enbpi)
        cross_EnbPI.append(res_cross_enbpi)

        print('\nEnCQR: ')
        encqr = EnCQR(model, n_ensemble, alpha_set, l_rate, max_epochs, batch_size=256, device=device, verbose=verbose)
        encqr.fit(X_train, Y_train)
        C_encqr = encqr.conformal(X_train, Y_train, X_test, Y_test)
        res_encqr, res_cross_encqr, C_encqr = transform(C_encqr, alpha_set, loader)
        metric_EnCQR.append(res_encqr)
        cross_EnCQR.append(res_cross_encqr)

    C_all = {'QR':np.array(C_QR), 'CQRS':np.array(C_CQRS), 'CQRA':np.array(C_CQRA), 'DCQRS':np.array(C_DCQRS),'DCQRA':np.array(C_DCQRA),
    'MCQRS':np.array(C_MCQRS), 'MCQRA':np.array(C_MCQRA), 'DMCQRS':np.array(C_DMCQRS), 'DMCQRA':np.array(C_DMCQRA),'EnbPI':C_enbpi, 'EnCQR':C_encqr}
    metric_all = {'QR':np.array(metric_QR), 'CQRS':np.array(metric_CQRS), 'CQRA':np.array(metric_CQRA), 'DCQRS':np.array(metric_DCQRS),'DCQRA':np.array(metric_DCQRA),
    'MCQRS':np.array(metric_MCQRS), 'MCQRA':np.array(metric_MCQRA), 'DMCQRS':np.array(metric_DMCQRS), 'DMCQRA':np.array(metric_DMCQRA),'EnbPI':np.array(metric_EnbPI), 'EnCQR':np.array(metric_EnCQR)}
    cross_all = {'QR':np.array(cross_QR), 'CQRS':np.array(cross_CQRS), 'CQRA':np.array(cross_CQRA), 'DCQRS':np.array(cross_DCQRS),'DCQRA':np.array(cross_DCQRA),
    'MCQRS':np.array(cross_MCQRS), 'MCQRA':np.array(cross_MCQRA), 'DMCQRS':np.array(cross_DMCQRS), 'DMCQRA':np.array(cross_DMCQRA),'EnbPI':np.array(cross_EnbPI), 'EnCQR':np.array(cross_EnCQR)}
    print('Done. ')

    return C_all, metric_all, cross_all

def extract(metric, num_alpha:int):
    '''
    将区间预测的评估结果对齐，将逐个conformal方法对比改为先逐个PINC level对比。
    Convert the metrics sequenced by the iter into metrics sequenced by confidence levels.
    
    input:
    metric: the metrics of prediction intervals, sequenced by the iter.
    '''
    
    A = np.array(metric)
    if A.ndim == 3:
        A = A.reshape((A.shape[0]*A.shape[1], A.shape[2]))
    num_model = int(A.shape[0] / num_alpha)
    out = []

    for i in range(num_alpha):
        for j in range(num_model):
            if i == 0:
                out.append(A[j*num_alpha, :])
            else:
                out.append(A[j*num_alpha+i, :])

    return np.array(out)

def plot_PI(PI, PINC, y_test_true, title:str, resultFolder:str, saveflag:False,ind_show=None, color='darkorange', figsize=(15,10), fontsize=20, lw=0.5):
    '''
    绘制预测区间图。

    input parameters:
    PI: the prediction intervals, ndarray or DataFrame, should be [N, len(PINC)*2].
    PINC: confidence levels of PIs, list or ndarray.
    y_test_true: the truth of the y_test, also the middle of the PIs, ndarray or DataFrame, [N, 1]
    subFolder: the name of the picture, str. 
    ind_show: the range of PIs to show. Example & default: ind_show = range(0, 200).
    color: the PIs' color, str. Example & default: color = 'darkorange'.
    
    '''

    PI = np.array(PI)
    y_test_true = np.array(y_test_true)
    print(f'PI.shape = {PI.shape}')
    print(f'PINC: {PINC}')
    assert PI.shape[1] == len(PINC)*2
    assert y_test_true.ndim < 3
    
    alphas = np.linspace(0.2, 1, len(PINC)) #透明度
    x = np.arange(len(PI))
    if ind_show is None:
        ind_show = range(200)

    plt.figure(figsize=figsize, dpi=60)
    plt.rc('font', family='Times New Roman', weight = 'medium', size=fontsize)
    plt.plot(x[ind_show], y_test_true[ind_show], 'k' ,label='Ground truth')
    for i in range(len(PINC)):
        plt.plot(x[ind_show], PI[ind_show, i], color=color, lw=lw) 
        plt.plot(x[ind_show], PI[ind_show, -(i+1)], color=color, lw=lw) 
        plt.fill_between(x[ind_show], PI[ind_show, i], PI[ind_show, -(i+1)], color=color, alpha=alphas[i],label='%d%% confidence PI'%(PINC[i]))

    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.title('PIs constructed by %s'%title)
    plt.tight_layout(pad=1)
    if saveflag:
        plt.savefig('{}{}.svg'.format(resultFolder, title))
        plt.savefig('{}{}.png'.format(resultFolder, title))
    else:
        plt.show()

def out_metric_cross(metric_all, cross_all, num_iter, PINC, resultFolder, saveflag):
    '''
    输出每个conformal方法的结果和区间交叉的检验结果。
    Output the PI metrics and boundary-crossing check results.

    input:
    metric_all: PIs' metrics, ndarray.
    cross_all: PIs' boundary-crossing check results, ndarray.

    '''

    columns_metric = ['MSE', 'MAE', 'RMSE', 'CRPS', 'SDE', 'PICP', 'MPIW', 'F', 'PINAW', 'ACE', 'CWC']
    columns_cross = ['MUCW', 'MLCW', 'CRPS', 'Cross loss']
    methods = ['QR', 'CQRS', 'CQRA', 'DCQRS', 'DCQRA', 'MCQRS', 'MCQRA', 'DMCQRS', 'DMCQRA', 'EnbPI', 'EnCQR']
    num_alpha = len(PINC)

    folder_metric = resultFolder + '/metrics/'
    folder_cross = resultFolder + '/cross/'

    if saveflag and not os.path.isdir(folder_metric):
        os.makedirs(folder_metric)
    if saveflag and not os.path.isdir(folder_cross):
        os.makedirs(folder_cross)

    index_metric = []
    for a in PINC:
        for i in range(num_iter):
            index_metric.append(str(int(a)))    
    print(index_metric)

    index_iter = [i+1 for i in range(num_iter)]
    print(index_iter)

    for i, key in enumerate(metric_all.keys()):
        m = extract(metric_all[key], num_alpha)
        m = pd.DataFrame(m, index=index_metric, columns=columns_metric)
        c = cross_all[key].squeeze().reshape((num_iter, len(columns_cross)))
        c = pd.DataFrame(c, index=index_iter, columns=columns_cross)
        if saveflag:
            m.to_csv(folder_metric+"%s.csv"%(methods[i]), index=True, header=True)
            c.to_csv(folder_cross+"%s.csv"%(methods[i]), index=True, header=True)

    print('Saved.')

def out_mean_std(metric_all, cross_all, PINC, num_iter, methods, resultFolder, saveflag):
    '''
    输出每个方法跑num_iter后的结果的均值和方差。
    Output the mean and standard deviation of PI metrics and boundary-crossing check results.
    '''

    num_alpha = len(PINC)
    out_all = []
    for key in metric_all.keys():
        out = []
        c = cross_all[key].squeeze()
        if c.ndim < 2:
            c = c.reshape((num_iter, len(c)))
        if num_iter == 1:
            ind_PICP = ['PICP', 'MPIW', 'CWC']
            columns_all = ind_PICP*num_alpha + ['MUCW', 'MLCW', 'Cross loss']
            for i in range(num_alpha):
                m = extract(metric_all[key], num_alpha)
                #['MSE', 'MAE', 'RMSE', 'CRPS', 'SDE', 'PICP', 'MPIW', 'F', 'PINAW', 'ACE', 'CWC']
                out.append(m[i*num_iter:(i+1)*num_iter, 5].mean(axis=0)) #average PICP
                out.append(m[i*num_iter:(i+1)*num_iter, 6].mean(axis=0)) #average MPIW
                out.append(m[i*num_iter:(i+1)*num_iter, -1].mean(axis=0)) #average CWC
            out.append(c[:, 0].mean(axis=0)) #average MUCW
            out.append(c[:, 1].mean(axis=0)) #average MLCW
            out.append(c[:, -1].mean(axis=0)) #average cross loss
            out_all.append(out)
        else:
            ind_PICP = ['PICP', 'PICP_std', 'MPIW', 'MPIW_std', 'CWC', 'CWC_std']
            columns_all = ind_PICP*num_alpha + ['MUCW', 'MUCW_std', 'MLCW', 'MLCW_std', 'Cross loss', 'Cross loss_std']
            for i in range(num_alpha):
                m = extract(metric_all[key], num_alpha)
                out.append(m[i*num_iter:(i+1)*num_iter, 5].mean(axis=0)) #average PICP
                out.append(m[i*num_iter:(i+1)*num_iter, 5].std(axis=0)) #std of the PICP
                out.append(m[i*num_iter:(i+1)*num_iter, 6].mean(axis=0)) #average MPIW
                out.append(m[i*num_iter:(i+1)*num_iter, 6].std(axis=0)) #std of the MPIW
                out.append(m[i*num_iter:(i+1)*num_iter, -1].mean(axis=0)) #average CWC
                out.append(m[i*num_iter:(i+1)*num_iter, -1].std(axis=0)) #std of the CWC
            out.append(c[:, 0].mean(axis=0)) #average MUCW
            out.append(c[:, 0].std(axis=0))  #std of the MUCW
            out.append(c[:, 1].mean(axis=0)) #average MLCW
            out.append(c[:, 1].std(axis=0))  #std of the MLCW
            out.append(c[:, -1].mean(axis=0)) #average cross loss
            out.append(c[:, -1].std(axis=0))  #std of the cross loss
            out_all.append(out)
            
    out_all = np.array(out_all)
    out_all = pd.DataFrame(out_all, index=methods, columns=columns_all)
    if saveflag:
        out_all.to_csv(resultFolder+"metrics_mean_std.csv", index=True, header=True)

    print('Saved.')

    return out_all
