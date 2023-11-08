import numpy as np
import torch.nn as nn
import torch
import time
import pandas as pd
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset, TensorDataset
from Losses import PinballLoss
# from torch.nn.utils.parametrizations import weight_norm  # torch==2.1.0+cu118
from torch.nn.utils import weight_norm  # torch==2.0+cu118


class power_data:
    def __init__(self, data_name, train_ratio=0.9, time_step=2,
                 start_num=5000, end_num=6000, x_size=3, **kwargs):
        data = pd.read_csv(data_name)
        data = data.dropna()
        data_np = np.array(data[['onpower', "ws0"]]
                           [start_num:(end_num + time_step)])
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

        self.y_train = data_1[time_step:(
            train_len + time_step), 0].reshape(-1, )
        self.y_test = data_1[(train_len + time_step):,
                             0].reshape(-1, ).reshape(-1, )
        self.y_train_yuan = data_np[time_step:(
            train_len + time_step), 0].reshape(-1, )
        self.y_test_yuan = data_np[(train_len + time_step):, 0].reshape(-1, )

        self.X_train = X[0:train_len, :]
        self.X_test = X[train_len:, :]
        self.data_max = data_max
        self.data_min = data_min

    def anti_mixmax(self, y):
        data_anti = y * (self.data_max[0] -
                         self.data_min[0]) + self.data_min[0]
        return data_anti

# 简单神经网络


class NET(nn.Module):
    def __init__(self, input_dim: int, hidden: int, out: int, activation=nn.Tanh, dropout=0.5):
        """ Initialization

        Parameters
        ----------

        input_dim : integer, input signal dimension (p)
        hidden : integer, hidden layer dimension
        out: integer, output layer dimension
        activation: string, activation function
        dropout : float, dropout rate

        """
        super(NET, self).__init__()
        self.input_dim = input_dim
        self.hidden = hidden
        self.out_dim = out
        self.activation = activation
        self.dropout = dropout

        # 激活函数选择
        if self.activation == 'relu':
            mid_act = torch.nn.ReLU()
        elif self.activation == 'tanh':
            mid_act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            mid_act = torch.nn.Sigmoid()
        elif self.activation == 'LeakyReLU':
            mid_act = torch.nn.LeakyReLU()
        elif self.activation == 'ELU':
            mid_act = torch.nn.ELU()
        elif self.activation == 'GELU':
            mid_act = torch.nn.GELU()

        self.model = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden),
            mid_act,
            nn.Dropout(self.dropout),
            #             nn.Linear(self.hidden, self.hidden),
            #             mid_act,
            #             nn.Dropout(self.dropout),
            nn.Linear(self.hidden, self.out_dim)
        )

    def init(self, W_var):
        # initialise weights
        for i, param in enumerate(self.model.parameters()):
            if isinstance(param, nn.Linear):
                param.data.normal_(0, W_var[i])

    def forward(self, x):
        out = self.model(x)
#         out = out.to(torch.float64)
        return out


class RNN(nn.Module):
    def __init__(self, x_size,
                 hidden_size,
                 output_dim,
                 activation='tanh',
                 device='cpu',
                 num_layers=1,
                 dropout=0.2):
        """ Initialization

        Parameters
        ----------
        quantiles : numpy array of quantile levels (q), each in the range (0,1)
        hidden_size : integer, hidden layer dimension
        dropout : float, dropout rate
        W_var: prior variance of weights and bias, ndarray, shape must equal to the number of parameters

        """
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.output_size = output_dim
        self.x_size = x_size
        self.device = device
        self.activation = activation
        # if activation == 'tanh':
        #     self.activation = activation
        # elif activation == 'relu':
        #     self.activation = activation
        # else:
        #     self.activation = 'tanh'
        self.dropout = nn.Dropout(dropout)

        self.model = torch.nn.Sequential(
            nn.RNN(input_size=self.x_size, hidden_size=self.hidden_size,
                   num_layers=self.num_layers, nonlinearity=self.activation, batch_first=True),
            nn.Linear(self.hidden_size, self.output_size)
        )

    def init(self, W_var):
        # initialise weights
        for i, param in enumerate(self.model.parameters()):
            param.data.normal_(0, W_var[i])

    def forward(self, x):
        '''
        x.shape: [batch_size, sequence length, input_dim]
        '''
        # 注意RNN模型的输入数据要变换格式
        # input shape should be (batch, sequence length, input size) when batch_first=True.
        # At this problem, sequence length = timestamp, inputsize = x_size.
        x = x.reshape(x.shape[0], self.x_size, -
                      1).transpose(1, 2)  # 先升维然后调换后两个维度顺序
        x = x.to(self.device)
        self.model = self.model.to(self.device)

        # h0 = Variable(torch.zeros(self.num_layers, x.shape[0], self.hidden_size)).to(self.device)
        # h0 = Variable(torch.zeros(self.num_layers, x.shape[0], self.hidden_size, device=self.device)).to(self.device)
        h0 = torch.zeros(
            self.num_layers, x.shape[0], self.hidden_size, device=self.device)
        ula, h_out = self.model[0](x, h0)
        out1 = self.dropout(ula[:, -1, :])
        out = self.model[1](out1)

        return out


class LSTM(nn.Module):
    def __init__(self, x_size,
                 hidden_size,
                 output_dim,
                 device='cpu',
                 num_layers=1,
                 dropout=0.2):
        """ Initialization

        Parameters
        ----------
        quantiles : numpy array of quantile levels (q), each in the range (0,1)
        hidden_size : integer, hidden layer dimension
        dropout : float, dropout rate
        W_var: prior variance of weights and bias, ndarray, shape must equal to the number of parameters

        """
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.output_size = output_dim
        self.x_size = x_size
        self.device = device
        self.dropout = nn.Dropout(dropout)
        self.model = torch.nn.Sequential(
            nn.LSTM(input_size=self.x_size, hidden_size=self.hidden_size,
                    num_layers=self.num_layers, batch_first=True),
            nn.Linear(self.hidden_size, self.output_size)
        )

    def init(self, W_var):
        # initialise weights
        for i, param in enumerate(self.model.parameters()):
            param.data.normal_(0, W_var[i])

    def forward(self, x):
        '''
        x.shape: [batch_size, sequence length, input_dim]
        '''
        # 注意LSTM模型的输入数据要变换格式
        # [batch, sequence length, input size] = [batch, timestamp, x_size]
        x = x.reshape(x.shape[0], self.x_size, -
                      1).transpose(1, 2)  # 先升维然后调换后两个维度顺序

        h_0 = torch.zeros(
            self.num_layers, x.shape[0], self.hidden_size, device=self.device)
        c_0 = torch.zeros(
            self.num_layers, x.shape[0], self.hidden_size, device=self.device)
        ula, (h_out, _) = self.model[0](x, (h_0, c_0))
        out1 = self.dropout(ula[:, -1, :])
        out = self.model[1](out1)

        return out


class GRU(nn.Module):
    def __init__(self, x_size,
                 hidden_size,
                 output_dim,
                 device='cpu',
                 num_layers=1,
                 dropout=0.2):
        """ Initialization

        Parameters
        ----------
        quantiles : numpy array of quantile levels (q), each in the range (0,1)
        hidden_size : integer, hidden layer dimension
        dropout : float, dropout rate
        W_var: prior variance of weights and bias, ndarray, shape must equal to the number of parameters

        """
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.output_size = output_dim
        self.x_size = x_size
        self.device = device
        self.dropout = nn.Dropout(dropout)
        self.model = torch.nn.Sequential(
            nn.GRU(input_size=self.x_size, hidden_size=self.hidden_size,
                   num_layers=self.num_layers, batch_first=True),
            nn.Linear(self.hidden_size, self.output_size)
        )

    def init(self, W_var):
        # initialise weights
        for i, param in enumerate(self.model.parameters()):
            param.data.normal_(0, W_var[i])

    def forward(self, x):
        '''
        x.shape: [batch_size, sequence length, input_dim]
        '''
        # 注意GRU模型的输入数据要变换格式
        # # [batch, sequence length, input size] = [batch, timestamp, x_size]
        x = x.reshape(x.shape[0], self.x_size, -
                      1).transpose(1, 2)  # 先升维然后调换后两个维度顺序

        h_0 = torch.zeros(
            self.num_layers, x.shape[0], self.hidden_size, device=self.device)
        ula, h_out = self.model[0](x, h_0)
        out1 = self.dropout(ula[:, -1, :])
        out = self.model[1](out1)

        return out


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        """
        其实这就是一个裁剪的模块，裁剪多出来的padding
        """
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        """
        相当于一个Residual block

        :param n_inputs: int, 输入通道数
        :param n_outputs: int, 输出通道数
        :param kernel_size: int, 卷积核尺寸
        :param stride: int, 步长，一般为1
        :param dilation: int, 膨胀系数
        :param padding: int, 填充系数
        :param dropout: float, dropout比率
        """
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        # 经过conv1，输出的size其实是(Batch, input_channel, seq_len + padding)
        self.chomp1 = Chomp1d(padding)  # 裁剪掉多出来的padding部分，维持输出时间步为seq_len
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)  # 裁剪掉多出来的padding部分，维持输出时间步为seq_len
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(
            n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        """
        参数初始化

        :return:
        """
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        """
        :param x: size of (Batch, input_channel, seq_len)
        :return:
        """
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        """
        TCN，目前paper给出的TCN结构很好的支持每个时刻为一个数的情况，即sequence结构，
        对于每个时刻为一个向量这种一维结构，勉强可以把向量拆成若干该时刻的输入通道，
        对于每个时刻为一个矩阵或更高维图像的情况，就不太好办。

        :param num_inputs: int， 输入通道数
        :param num_channels: list，每层的hidden_channel数，例如[25,25,25,25]表示有4个隐层，每层hidden_channel数为25
        :param kernel_size: int, 卷积核尺寸
        :param dropout: float, drop_out比率
        """
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i   # 膨胀系数：1，2，4，8……
            # 确定每一层的输入通道数
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]  # 确定每一层的输出通道数
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        输入x的结构不同于RNN，一般RNN的size为(Batch, seq_len, channels)或者(seq_len, Batch, channels)，
        这里把seq_len放在channels后面，把所有时间步的数据拼起来，当做Conv1d的输入尺寸，实现卷积跨时间步的操作，
        很巧妙的设计。

        :param x: size of (Batch, input_channel, seq_len)
        :return: size of (Batch, output_channel, seq_len)
        """
        return self.network(x)


class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout, device='cuda'):
        super(TCN, self).__init__()
        self.input_size = input_size
        self.device = device
        self.tcn = TemporalConvNet(
            self.input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def init(self, W_var):
        self.linear.weight.data.normal_(0, 0.01)

    def forward(self, x):
        x = x.reshape(x.shape[0], self.input_size, -1)
        x = x.to(self.device)
        y1 = self.tcn(x)
        return self.linear(y1[:, :, -1])


class RegressionEstimator():
    def __init__(self, model, crit, max_epochs, batch_size, device, optimizer, scheduler, verbose=False, logger=None):

        self.model = model
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.device = device
        # control whether output the training process, bool.
        self.verbose = verbose
        self.crit = crit  # loss function
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.logger = logger

    def fit(self, X_train, Y_train, X_valid=None, Y_valid=None):

        train_data = TensorDataset(X_train, Y_train)
        train_dataloader = DataLoader(
            dataset=train_data, batch_size=self.batch_size, shuffle=False)

        model = self.model.to(self.device)
        optimizer = self.optimizer
        scheduler = self.scheduler
        train_loss_history = []
        validation_loss_history = []

        for epoch in range(self.max_epochs):
            start_time = time.time()
            loss_all = []
            model.train()

            for data in train_dataloader:

                x, y = data
                x = x.to(self.device)
                y = y.to(self.device)

                optimizer.zero_grad()
                out = model(x)
                loss = self.crit(out, y)
                loss.requires_grad_(True)
                loss.backward()

                optimizer.step()
                loss_all.append(loss.item())

            scheduler.step()
            end_time = time.time()
            cost_time = end_time - start_time

            train_loss = np.mean(np.array(loss_all))
            train_loss_history.append(train_loss)

            # '---------------evaluating model on validation set------------------'
            if X_valid is not None:
                model.eval()
                valid_data = TensorDataset(X_valid, Y_valid)
                validation_dataloader = DataLoader(
                    dataset=valid_data, batch_size=self.batch_size, shuffle=False)
                loss_all = []
                with torch.no_grad():
                    for data in validation_dataloader:
                        x, y = data
                        x = x.to(self.device)
                        y = y.to(self.device)
                        output = model(x)
                        loss = self.crit(output, y)
                        loss_all.append(loss.item())

                validation_loss = np.mean(np.array(loss_all))
                validation_loss_history.append(validation_loss)
                if self.verbose and (epoch+1) % 100 == 0:
                    if self.logger:
                        self.logger.info('Epoch:{:d}, train_loss: {:.4f}, validation_loss: {:.4f}, cost_time: {:.2f}s'
                                         .format(epoch+1, train_loss, validation_loss, cost_time))
                    else:
                        print('Epoch:{:d}, train_loss: {:.4f}, validation_loss: {:.4f}, cost_time: {:.2f}s'
                              .format(epoch+1, train_loss, validation_loss, cost_time))
            else:
                if self.verbose and (epoch+1) % 100 == 0:
                    print('Epoch:{:d}, train_loss: {:.4f}, cost_time: {:.2f}s'
                          .format(epoch+1, train_loss, cost_time))

        return train_loss_history, validation_loss_history

    def predict(self, x):

        model = self.model.to(self.device)
        model.eval()
        with torch.no_grad():
            x = x.to(self.device)
            pred = model(x)

        res = pred.data.cpu().numpy()

        return res


class QuantileRegressionEstimator():
    def __init__(self, model, alpha_set, max_epochs, batch_size, device, l_rate=0.001, verbose=False, logger=None):

        self.model = model  # the list of models
        self.alpha_set = alpha_set
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.device = device
        self.l_rate = l_rate  # learning rate
        # control whether output the training process, bool.
        self.verbose = verbose
        self.num_alpha = len(self.alpha_set)
        self.quantiles = np.zeros(2*self.num_alpha)
        for i in range(self.num_alpha):
            q_low = self.alpha_set[i] / 2
            q_high = 1 - q_low
            self.quantiles[i] = q_low
            self.quantiles[-(i+1)] = q_high

        self.q_num = len(self.quantiles)
        self.crit = PinballLoss(quantiles=self.quantiles)  # loss function
        self.logger = logger

    def fit(self, X_train, Y_train, X_valid=None, Y_valid=None):

        train_data = TensorDataset(X_train, Y_train)
        train_dataloader = DataLoader(
            dataset=train_data, batch_size=self.batch_size, shuffle=False)

        model = self.model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.l_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=100, gamma=0.9)  # 动态学习率调整
        train_loss_history = []
        validation_loss_history = []

        for epoch in range(self.max_epochs):
            start_time = time.time()
            loss_all = []
            model.train()

            for data in train_dataloader:

                x, y = data
                x = x.to(self.device)
                y = y.to(self.device)

                optimizer.zero_grad()

                out = model(x).to(self.device)
                # print(f'out.device: {out.device}, y.device: {y.device}')
                loss = self.crit(out, y)
                loss.requires_grad_(True)
                loss.backward()

                optimizer.step()
                loss_all.append(loss.item())

            scheduler.step()
            end_time = time.time()
            cost_time = end_time - start_time

            train_loss = np.mean(np.array(loss_all))
            train_loss_history.append(train_loss)

            # '---------------evaluating model on validation set------------------'
            if X_valid is not None:
                model.eval()
                valid_data = TensorDataset(X_valid, Y_valid)
                validation_dataloader = DataLoader(
                    dataset=valid_data, batch_size=self.batch_size, shuffle=False)
                loss_all = []
                with torch.no_grad():
                    for data in validation_dataloader:
                        x, y = data
                        x = x.to(self.device)
                        y = y.to(self.device)
                        output = model(x)
                        loss = self.crit(output, y)
                        loss_all.append(loss.item())

                validation_loss = np.mean(np.array(loss_all))
                validation_loss_history.append(validation_loss)
                if self.verbose and (epoch+1) % 100 == 0:
                    if self.logger:
                        self.logger.info('Epoch:{:d}, train_loss: {:.4f}, validation_loss: {:.4f}, cost_time: {:.2f}s'
                                         .format(epoch+1, train_loss, validation_loss, cost_time))
                    else:
                        print('Epoch:{:d}, train_loss: {:.4f}, validation_loss: {:.4f}, cost_time: {:.2f}s'
                              .format(epoch+1, train_loss, validation_loss, cost_time))
            else:
                if self.verbose and (epoch+1) % 100 == 0:
                    print('Epoch:{:d}, train_loss: {:.4f}, cost_time: {:.2f}s'
                          .format(epoch+1, train_loss, cost_time))

        return train_loss_history, validation_loss_history

    def predict(self, x):

        self.model.eval()
        with torch.no_grad():
            x = x.to(self.device)
            pred = self.model(x)

        res = pred.data.cpu().numpy()

        return res


class EnCQR():
    def __init__(self, model, n_ensemble, alpha_set, l_rate: float, max_epochs: int, batch_size: int, device='cuda', verbose=True):
        assert n_ensemble > 1  # 等于1会报错，难以划分 s_b 和 no_s_b
        self.model = model  # 集成学习模型的基学习器
        self.n_ensemble = n_ensemble  # 基学习器数量
        self.NNs = []
        for _ in range(self.n_ensemble):
            self.NNs.append(self.model)
        self.l_rate = l_rate  # 学习率
        self.max_epochs = max_epochs
        self.batch_size = batch_size  # 越大更新越慢，int.
        self.device = device
        self.verbose = verbose  # 是否输出中间过程
        self.alpha_set = alpha_set
        self.num_alpha = len(alpha_set)

    def fit(self, X_train, Y_train):

        T = X_train.shape[0]
        tb = int(np.floor(T / self.n_ensemble))

        S = np.arange(0, T)
        for b in range(self.n_ensemble):
            print('-- training: ' + str(b+1) + ' of ' +
                  str(self.n_ensemble) + ' NNs --')

            s_b = range(tb*b, (tb*b+tb))
            no_s_b = np.delete(S, s_b, 0)
            x_s_b, y_s_b = X_train[s_b, :], Y_train[s_b].reshape(len(s_b), 1)

            x_no_s_b = X_train[no_s_b, :]
            y_no_s_b = Y_train[no_s_b].reshape(len(no_s_b), 1)

            if self.verbose:
                print(
                    f'x_s_b.shape = {x_s_b.shape}, y_s_b = {y_s_b.shape}, x_no_s_b.shape = {x_no_s_b.shape}, y_no_s_b.shape = {y_no_s_b.shape}')

            learner = QuantileRegressionEstimator(
                self.NNs[b], self.alpha_set, self.max_epochs, self.batch_size, self.device, self.l_rate, self.verbose)
            learner.fit(x_s_b, y_s_b, x_no_s_b, y_no_s_b)
            print('model: %d finished training.' % (b+1))

        self.no_s_b = no_s_b

    def predict(self, x):
        '''
        回归预测。Point forecasting.

        out:
        res: point forecasting results of x, ndarray, [N, ].
        '''
        n_ensemble = len(self.NNs)
        P = torch.zeros(n_ensemble, x.shape[0], self.num_alpha*2)

        for b in range(n_ensemble):

            model = self.NNs[b]
            model.eval()
            with torch.no_grad():
                x = x.to(self.device)
                pred = model(x)

            P[b, :, :] = pred.to(torch.float32)

        res = P.mean(axis=0)
        res = res.numpy()
        res = res.squeeze()

        return res

    def conformal(self, X_train, Y_train, X_test, Y_test, step=None):
        '''
        区间预测。Interval prediction. fit完直接就可以调用来构造预测区间。

        out:
        C: prediction intervals.
        '''

        Y_train, Y_test = np.array(Y_train), np.array(Y_test)

        res_train = self.predict(X_train)
        res_test = self.predict(X_test)
        T1 = res_test.shape[0]
        if step == None:
            step = self.batch_size

        # Initialize the asymmetric conformity scores.
        C = np.zeros((T1, self.num_alpha*2))
        E_low, E_high = np.zeros((len(self.no_s_b), self.num_alpha)), np.zeros(
            (len(self.no_s_b), self.num_alpha))
        Q_low, Q_high = np.zeros(
            (self.num_alpha,)), np.zeros((self.num_alpha,))
        for i in range(self.num_alpha):
            E_low[:, i] = (res_train[self.no_s_b, i].reshape(
                (len(self.no_s_b), 1)) - Y_train[self.no_s_b].reshape((len(self.no_s_b), 1))).squeeze()
            E_high[:, -(i+1)] = (Y_train[self.no_s_b].reshape((len(self.no_s_b), 1)) -
                                 res_train[self.no_s_b, -(i+1)].reshape((len(self.no_s_b), 1))).squeeze()

        # Comformalize the prediction intervals.
        for t in range(T1):
            for i, alpha in enumerate(self.alpha_set):

                Q_low[i] = np.quantile(E_low[:, i], 1 - alpha / 2)
                Q_high[-(i+1)] = np.quantile(E_high[:, -(i+1)], 1 - alpha / 2)

                C[t, i] = res_test[t, i] - Q_low[i]
                C[t, -(i+1)] = res_test[t, -(i+1)] + Q_high[-(i+1)]

            # Update the lists of conformity scores
            if t % step == 0 and step < T1:
                # print('t = %d, Q_low[0] = %f, Q_high[-1] = %f, E_low.shape = %s, E_high.shape = %s.' %
                #   (t,Q_low[0],Q_high[-1],str(E_low.shape), str(E_high.shape)))
                for j in range(t - step, t-1):
                    for i in range(self.num_alpha):
                        e_low = res_test[j, i] - Y_test[j]
                        e_high = Y_test[j] - res_test[j, -(i+1)]
                        E_low_temp = np.delete(E_low[:, i], 0, 0)  # 删除第一个元素
                        E_low_temp = np.append(E_low_temp, e_low)  # 添加新的元素
                        E_low[:, i] = E_low_temp
                        E_high_temp = np.delete(E_high[:, -(i+1)], 0, 0)
                        E_high_temp = np.append(E_high_temp, e_high)
                        E_high[:, -(i+1)] = E_high_temp

        return C
