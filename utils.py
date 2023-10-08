import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import scienceplots
from sklearn.preprocessing import StandardScaler

class power_data:
    def __init__(self, data_name, train_ratio=0.9, time_step=2,
                 start_num=5000, end_num=6000, x_size=3, **kwargs):
        data = pd.read_csv(data_name)
        data = data.dropna()
        data_np = np.array(data[['onpower', "ws0"]][start_num:(end_num + time_step)])
        data_min = data_np.min(axis=0)
        data_max = data_np.max(axis=0)
        data_1 = (data_np - data_min) / (data_max - data_min)
        data_1[data_1 == 0] = 0.00001
        data_1[data_1 == 1] = 0.99999
        data_len = end_num - start_num
        train_len = round(data_len * train_ratio)
        X = np.zeros(shape=(end_num - start_num, x_size * time_step))

        for i in range(time_step):
            X[:, i] = data_1[i:(end_num - start_num + i), 0]
        for i in range(1, x_size):
            for j in range(time_step):
                X[:, (i * time_step + j)] = data_1[j:(end_num - start_num + j), i]

        self.y_train = data_1[time_step:(train_len + time_step), 0].reshape(-1, )
        self.y_test = data_1[(train_len + time_step):, 0].reshape(-1, ).reshape(-1, )
        self.y_train_yuan = data_np[time_step:(train_len + time_step), 0].reshape(-1, )
        self.y_test_yuan = data_np[(train_len + time_step):, 0].reshape(-1, )

        self.X_train = X[0:train_len, :]
        self.X_test = X[train_len:, :]
        self.data_max = data_max
        self.data_min = data_min

    def anti_mixmax(self, y):
        data_anti = y * (self.data_max[0] - self.data_min[0]) + self.data_min[0]
        return data_anti

class TimeSeriesDataLoader:
    def __init__(self, data, window_size, label_column, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, scaler=None):
        self.data         = data
        self.window_size  = window_size
        self.label_column = label_column
        self.train_ratio  = train_ratio
        self.val_ratio    = val_ratio
        self.test_ratio   = test_ratio
        self.scaler       = scaler

        self.train_X, self.train_y = None, None
        self.val_X, self.val_y = None, None
        self.test_X, self.test_y = None, None

        self._prepare_data()

    def _prepare_data(self):
        # 删除缺失值
        self.data = self.data.dropna()

        # 划分特征和目标变量
        X = self.data.drop(columns=[self.label_column]).values
        y = self.data[self.label_column].values

        # 处理缺失值（如果有）
        # 进行标准化（如果需要）
        if self.scaler is None:
            self.scaler_x = StandardScaler()
            self.scaler_y = StandardScaler()
            X = self.scaler_x.fit_transform(X)
            y = self.scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

        data = np.concatenate((X, y.reshape(-1, 1)), axis=1)
        # 生成窗口数据
        sequences_X, sequences_y = [], []
        for i in range(len(self.data) - self.window_size + 1):
            window_X = data[i:i + self.window_size]
            window_y = y[i + self.window_size - 1]  # 取窗口最后一个样本作为目标变量
            sequences_X.append(window_X)
            sequences_y.append(window_y)
        sequences_X, sequences_y = np.array(sequences_X), np.array(sequences_y)
        sequences_X = sequences_X.reshape(len(sequences_X), self.window_size*sequences_X.shape[2])

        # 划分训练集、验证集、测试集
        train_size = int(len(sequences_X) * self.train_ratio)
        val_size = int(len(sequences_X) * self.val_ratio)
        test_size = len(sequences_X) - train_size - val_size

        self.train_X, self.train_y = sequences_X[:train_size], sequences_y[:train_size]
        self.val_X, self.val_y = sequences_X[train_size:train_size + val_size], sequences_y[train_size:train_size + val_size]
        self.test_X, self.test_y = sequences_X[train_size + val_size:], sequences_y[train_size + val_size:]

    def inverse_transform(self, data, is_label=True):
        # 将标准化后的数据还原
        if is_label:
            if data.ndim < 2:
                return self.scaler_y.inverse_transform(data.reshape(-1, 1)).flatten()
            else:
                for i in range(data.shape[1]):
                    data[:, i] = self.scaler_y.inverse_transform(data[:, i].reshape(-1, 1)).flatten()
                return data
        else:
            return self.scaler_x.inverse_transform(data)

    def get_train_data(self, to_tensor=False):
        if to_tensor:
            return torch.from_numpy(self.train_X).to(torch.float32), torch.from_numpy(self.train_y).to(torch.float32)
        else:
            return self.train_X, self.train_y

    def get_val_data(self, to_tensor=False):
        if to_tensor:
            return torch.from_numpy(self.val_X).to(torch.float32), torch.from_numpy(self.val_y).to(torch.float32)
        else:
            return self.val_X, self.val_y

    def get_test_data(self, to_tensor=False):
        if to_tensor:
            return torch.from_numpy(self.test_X).to(torch.float32), torch.from_numpy(self.test_y).to(torch.float32)
        else:
            return self.test_X, self.test_y

def asym_nonconformity(label, low, high):
    """
    Compute the asymetric conformity score
    """
    error_high = label - high 
    error_low = low - label
    return error_low, error_high

def plot_PI(PI, PINC, y_test_true, title:str, resultFolder:str, saveflag:False,ind_show=None, color='darkorange', 
            style=['science'], figsize=(15,10), fontsize=20, lw=0.5, save_dpi=300):
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

    with plt.style.context(style):
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
        # plt.title('PIs constructed by %s'%title)
        plt.tight_layout(pad=1)
        if saveflag:
            plt.savefig('{}_{}.svg'.format(resultFolder, title), dpi=save_dpi, bbox_inches='tight')
            plt.savefig('{}_{}.png'.format(resultFolder, title), dpi=save_dpi, bbox_inches='tight')
        else:
            plt.show()
