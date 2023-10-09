import torch, os
import torch.nn as nn
import numpy as np
import pandas as pd
from models import QuantileRegressionEstimator, RegressionEstimator
from Losses import PinballLoss

class NESCQR:
    def __init__(self, model_pool:list, label_pool:list, batch_size:int, M:int, alpha_set:list, 
                 l_rate:float, max_epochs:int, replace, symmetric, saveflag, save_dir, 
                 alpha_base=None, step=2, device='cuda', verbose=True):
        assert 0 < M <= len(model_pool), "M must be in range (0, len(model_pool)]"
        self.model_pool  = model_pool
        self.label_pool  = label_pool  # 与model_pool里每个模型一一对应的模型名字
        self.batch_size  = batch_size
        self.M           = M           # 最终的集成模型的基学习器个数
        self.alpha_set   = alpha_set   # 置信水平集合
        self.l_rate      = l_rate      # 学习率
        self.max_epochs  = max_epochs
        self.device      = device
        self.saveflag   = saveflag
        self.save_dir    = save_dir
        self.alpha_base  = alpha_base if alpha_base else max(alpha_set)
        self.quantiles   = [self.alpha_base / 2, 1 - self.alpha_base / 2]
        self.loss_fn     = PinballLoss(self.quantiles, self.device)
        self.replace     = replace    # 是否有放回地前向选择
        self.step        = step       # DMCQR算法更新步长，int, 越小更新越快越准确
        self.symmetric   = symmetric  # 是否采用对称性conformity score
        # self.logger      = logger
        self.verbose     = verbose
        
    def init_training(self, X_train, Y_train, X_val, Y_val, saveflag=False):
        """
        先训练好每个基学习器. Initialize each base leaner in the model pool.

        Args:
            X_train (array-like): The training data. Tensor.
            Y_train (array-like): The training labels. Tensor.
            X_val (array-like): The validation data. Tensor.
            Y_val (array-like): The validation labels. Tensor.
            saveflag (bool, optional): Whether to save the trained models. Defaults to False.
        Returns:
            list: The list of trained models.
        """

        assert len(X_train) == len(Y_train)
        assert len(X_val)   == len(Y_val)

        print(f'X_train.shape: {X_train.shape}, Y_train.shape: {Y_train.shape}')
        print(f'X_val.shape: {X_val.shape}, Y_val.shape: {Y_val.shape}')

        num_models = len(self.model_pool)
        model_pool_trained = []
        for i, model in enumerate(self.model_pool):
            print(f'Model {i+1}/{num_models} {self.label_pool[i]} starts training...')

            # 采用DMCQR得到最终的预测区间，则只需要最大的alpha，即两条分位数即可得到多条预测区间上下界。
            learner = QuantileRegressionEstimator(model, [self.alpha_base], self.max_epochs,
                                                   self.batch_size,self.device, self.l_rate, self.verbose)
            learner.fit(X_train, Y_train, X_val, Y_val)
            model_pool_trained.append(learner)
            print(f'Model {i+1}/{num_models} {self.label_pool[i]} finished training.')
            
            if saveflag:
                torch.save(learner, f'{self.save_dir}/trained_models/{self.label_pool[i]}.pth')
                print(f'Model {i+1}/{num_models} saved.')

        return model_pool_trained

    def forward_selection(self, X_val, Y_val, model_pool_trained, label_pool, replace=True):
        # 前向选择出最优集成模型组合
        """
        前向选择出最优集成模型组合. 
        Find the best combination of the ensemble model through forward selection.

        Args:
            X_val (torch.Tensor): Validation data. Tensor.
            Y_val (torch.Tensor): Validation labels. Tensor.
            model_pool_trained (list): List of trained models.
            label_pool (list): List of labels for the models.
            replace (bool, optional): Whether to use replacement during selection. Defaults to True.
            
        Returns:
            tuple: Tuple containing the selected models and their corresponding labels.
        """

        X_val  , Y_val   = X_val.to(self.device), Y_val.to(self.device)
        # pool = dict(zip(label_pool, model_pool_trained))
        if replace:
            print('Forward selection with replacement.')
        else:
            print('Forward selection without replacement.')

        selected_model, selected_label = [], []
        while len(selected_model) < self.M:
            best_loss = np.inf
            
            for i in range(len(model_pool_trained)):
                models_ = selected_model + [model_pool_trained[i]]
                merged_output = torch.stack([torch.from_numpy(model.predict(X_val)) for model in models_])
                merged_output = torch.mean(merged_output, axis=0)
                loss = self.loss_fn(merged_output, Y_val)
                if loss.item() < best_loss:
                    best_model = model_pool_trained[i]
                    best_loss  = loss.item()
                    best_label = i

            selected_model.append(best_model)
            selected_label.append(label_pool[best_label])
            if not replace:  # 无放回
                model_pool_trained.pop(best_label)
                label_pool.pop(best_label)
                
        print(f'Model selected: {selected_label}')

        return selected_model, selected_label

    def conformal(self, res_val, Y_val, res_test, Y_test, step, symmetric=True):
        # DMCQR
        assert res_val.shape[0] == Y_val.shape[0]
        assert res_val.shape[1] == 2
        assert res_test.shape[1] == 2

        Y_val  , Y_test   = np.array(Y_val),   np.array(Y_test)
        res_val, res_test = np.array(res_val), np.array(res_test)
        Y_all     = np.concatenate((Y_val, Y_test), axis=0)
        res_all   = np.concatenate((res_val, res_test), axis=0)
        num_alpha = len(self.alpha_set)
        conf_PI   = np.zeros((len(res_test), num_alpha*2))
        val_size  = len(Y_val)
        test_size = len(Y_test)

        if symmetric:  
            # DMCQRS, use symmetric conformity score, 对称误差集合
            print('Use symmetric conformity score to calibrate quantiles.')
            E = list(np.max((res_val[:, 0] - Y_val, Y_val - res_val[:, -1]), axis=0))  # 误差集合，队列，先进先出
            Q = np.zeros(num_alpha)

            for t in range(val_size, val_size + test_size):
                for i, alpha in enumerate(self.alpha_set):
                    Q[i] = np.quantile(E, (1-alpha)*(1+1/val_size))
                    conf_PI[:, i] = res_test[:, 0] - Q[i]
                    conf_PI[:, -(i+1)] = res_test[:, -1] + Q[i]

                    if t % step == 0:
                        # print(f't = {t}, Q = {Q}')
                        for j in range(t - self.step, t - 1):
                            e = np.max((res_all[j, 0] - Y_all[j], Y_all[j] - res_all[j, -1]),axis=0)
                            E.pop(0)
                            E.append(e)   

            return conf_PI
        
        else:   
            # DMCQRS, use asymmetric conformity score, 非对称误差集合
            print('Use asymmetric conformity score to calibrate quantiles.')
            Q_low, Q_high = np.zeros(num_alpha), np.zeros(num_alpha)
            E_low = list(res_val[:, 0] - Y_val)    # 下界误差集合
            E_high = list(Y_val - res_val[:,-1])   # 上界误差集合
                
            for t in range(val_size, val_size + test_size):
                for i, alpha in enumerate(self.alpha_set):
                    Q_low[i] = np.quantile(E_low, (1 - alpha / 2))
                    Q_high[-(i+1)] = np.quantile(E_high, (1 - alpha / 2))
                    
                    conf_PI[:, i] = res_test[:, 0] - Q_low[i]
                    conf_PI[:, -(i+1)] = res_test[:, -1] + Q_high[-(i+1)]

                if t % step == 0:
                    # print(f't: {t}, Q_low: {Q_low}, Q_high: {Q_high}')
                    for j in range(t - step, t - 1):
                        e_low = res_all[j, 0] - Y_all[j]
                        e_high = Y_all[j] - res_all[j, -1]
                        E_low.pop(0)
                        E_low.append(e_low)
                        E_high.pop(0)
                        E_high.append(e_high)

            return conf_PI     

    def fit(self):
        model_pool_trained = self.init_training()
        self.model_pool_selected, self.selected_label = self.forward_selection(model_pool_trained, self.label_pool, self.replace)

    def predict(self, X_val, Y_val, X_test, Y_test):
        # construct prediction intervals
        # X_val  , Y_val   = self.data_loader.get_val_data(to_tensor=True)
        # if not X_test:
        #     X_test,  Y_test  = self.data_loader.get_test_data(to_tensor=True)
        Y_val  , Y_test  = Y_val.detach().numpy(), Y_test.detach().numpy()
        X_val   = X_val.to(self.device)
        X_test  = X_test.to(self.device)

        # pred = self.model_pool_selected[0].predict(X_val)
        # # print(f'pred.shape: {pred.shape}')
        res_val = torch.stack([torch.from_numpy(model.predict(X_val)) for model in self.model_pool_selected])
        res_val = torch.mean(res_val, axis=0)
        res_val = res_val.detach().numpy()
        res_test = torch.stack([torch.from_numpy(model.predict(X_test)) for model in self.model_pool_selected])
        res_test = torch.mean(res_test, axis=0)
        res_test = res_test.detach().numpy()
        # print(f'res_val.shape: {res_val.shape}, res_test.shape: {res_test.shape}')

        self.conf_PI = self.conformal(res_val, Y_val, res_test, Y_test, self.step, self.symmetric)
        # if inverse_normalization:  # 逆标准化回原来的量纲
        #     self.conf_PI = self.data_loader.inverse_transform(self.conf_PI, is_label=True)

        cols = [str(round(alpha/2, 3)) for alpha in self.alpha_set] + [str(round(1-alpha/2, 3)) for alpha in reversed(self.alpha_set)]
        if self.saveflag:
            df = pd.DataFrame(self.conf_PI, columns=cols)
            df.to_csv(os.path.join(self.save_dir,'conf_PIs.csv'), index=False)
            
        return self.conf_PI
    

class EnbPI():
    def __init__(self, NNs:list, alpha_set, l_rate:float, max_epochs:int, batch_size:int, device='cuda', verbose=True):
        self.NNs = NNs   #集成学习模型，ensemble model, list.
        self.crit = nn.MSELoss()  #loss function
        self.l_rate = l_rate  #学习率
        self.max_epochs = max_epochs
        self.batch_size = batch_size  #越大更新越慢，int.
        self.device = device
        self.verbose = verbose  #是否输出中间过程
        self.alpha_set = alpha_set

    def fit(self, X_train, Y_train):
        '''
        将EnbPI拆分，此函数为回归学习器。
        先回归，然后conformal得到预测区间上下界（均值加减）。

        input:
        X_train: torch.Tensor, training data. 
        Y_train: torch.Tensor, training labels. 

        '''

        train_size = X_train.shape[0]
        n_ensemble = len(self.NNs)
        S = np.arange(0, train_size)

        for b in range(n_ensemble):
            print('-- EnbPI training: ' + str(b+1) + ' of ' + str(n_ensemble) + ' NNs --')

            s_b = np.random.choice(range(0, train_size), size=train_size, replace=True)
            self.no_s_b = np.delete(S, s_b, 0)  #不在s_b子集的序号。
            x_s_b = X_train[s_b, :]
            y_s_b = Y_train[s_b].reshape(train_size, 1)

            x_no_s_b = X_train[self.no_s_b, :]
            y_no_s_b = Y_train[self.no_s_b].reshape(len(self.no_s_b), 1)
            
            if self.verbose:
                print(f'x_s_b.shape = {x_s_b.shape}, y_s_b = {y_s_b.shape}, x_no_s_b.shape = {x_no_s_b.shape}, y_no_s_b.shape = {y_no_s_b.shape}')

            model = self.NNs[b]
            optimizer = torch.optim.Adam(model.parameters(), lr=self.l_rate)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)  #动态学习率调整

            learner = RegressionEstimator(model, self.crit, self.max_epochs, self.batch_size, self.device, optimizer, scheduler, self.verbose)
            learner.fit(x_s_b, y_s_b, x_no_s_b, y_no_s_b)
            print('model: %d finished training.' % (b+1))  

    def predict_point(self, x):
        '''
        This function performs point forecasting.

        Args:
        x: input data, ndarray, [N, ].

        Returns:
        res: point forecasting results of x, ndarray, [N, ].
        '''
        n_ensemble = len(self.NNs)
        P = torch.zeros(n_ensemble, x.shape[0], 1, dtype=torch.float32, device=self.device)

        for b in range(n_ensemble):
            model = self.NNs[b]
            model.eval()
            with torch.no_grad():
                pred = model(x.to(self.device))

            P[b, :, :] = pred
        
        res = P.mean(axis=0).squeeze().cpu().numpy()

        return res

    def predict_interval(self, X_train, Y_train, X_test, Y_test, step=None):
        '''
        区间预测。Interval prediction. fit后方可调用。
        This function performs interval prediction based on the trained model.
            
        Args:
            X_train: The input features for training.
            Y_train: The target values for training.
            X_test: The input features for testing.
            Y_test: The target values for testing.
            step: The update speed of the conformity score. Smaller values make the update faster. 
                Defaults to None, in which case it is set to the batch size.
        
        Returns:
            C: The prediction intervals.
        '''
        
        Y_train, Y_test = np.array(Y_train), np.array(Y_test)
        num_alpha = len(self.alpha_set)
        if step == None:
            step = self.batch_size

        res_train = self.predict_point(X_train)
        res_test = self.predict_point(X_test)
        test_size = res_test.shape[0]

        C = np.zeros((test_size, num_alpha*2))
        Q = np.zeros(num_alpha) 
        
        for i, alpha in enumerate(self.alpha_set):
            E = abs(res_train[self.no_s_b].reshape((len(self.no_s_b),1)) - Y_train[self.no_s_b].reshape((len(self.no_s_b),1)))
            for t in range(test_size):
                Q[i] = np.quantile(E, 1 - alpha)
                C[t, i] = res_test[t] - Q[i]
                C[t, -(i+1)] = res_test[t] + Q[i]

                if t % step == 0:
                    # print('t = %d, alpha = %.2f, Q[0] = %.4f, Q[1] = %.4f, Q[2] = %.4f, E.shape = %s' % (t,
                    #          alpha, Q[0], Q[1], Q[2], str(E.shape)))
                    for j in range(t - step, t-1):
                        e = abs(Y_test[j] - res_test[j])
                        E = np.delete(E, 0, 0)      #删除第一个元素
                        E = np.append(E, e)         #添加新的元素
             
        return C

# def EnCQR(train_data, val_x, val_y, test_x, test_y, P):
#     """
#     Parameters
#     ----------
#     train_data : list of data to train an ensemble of models
#     test_x : input test data
#     test_y : output test data
#     P : dictionary of parameters

#     Returns
#     -------
#     PI : original PI produced by the ensemble model
#     conf_PI : PI after the conformalization
#     """

#     index = np.arange(P['B'])
#     s = P['time_steps_out']
    
#     # dict containing LOO predictions
#     dct_lo = {}
#     dct_hi = {}
#     for key in index:
#       dct_lo['pred_%s' % key] = []
#       dct_hi['pred_%s' % key] = []
    
#     # training a model for each sub set Sb
#     ensemble_models = []
#     for b in range(P['B']):
#         f_hat_b = regression_model(P)
#         f_hat_b.fit(train_data[index[b]][0], train_data[index[b]][1], val_x, val_y)
#         ensemble_models.append(f_hat_b)
        
#         # Leave-one-out predictions for each Sb
#         indx_LOO = index[np.arange(len(index))!=b]
#         for i in range(len(indx_LOO)):
#             pred = f_hat_b.transform(train_data[indx_LOO[i]][0])
#             dct_lo['pred_%s' %indx_LOO[i]].append(pred[:,:,0])
#             dct_hi['pred_%s' %indx_LOO[i]].append(pred[:,:,2])
            
#     f_hat_b_agg_low  = np.zeros((train_data[index[0]][0].shape[0], P['time_steps_out'], P['B']))
#     f_hat_b_agg_high = np.zeros((train_data[index[0]][0].shape[0], P['time_steps_out'], P['B']))
#     for b in range(P['B']):
#         f_hat_b_agg_low[:,:,b] = np.mean(dct_lo['pred_%s' %b],axis=0) 
#         f_hat_b_agg_high[:,:,b] = np.mean(dct_hi['pred_%s' %b],axis=0)  
        
#     # residuals on the training data
#     epsilon_low = []
#     epsilon_hi=[]
#     for b in range(P['B']):
#         e_low, e_high = utils.asym_nonconformity(label=train_data[b][1], 
#                                                   low=f_hat_b_agg_low[:,:,b], 
#                                                   high=f_hat_b_agg_high[:,:,b])
#         epsilon_low.append(e_low)
#         epsilon_hi.append(e_high)
#     epsilon_low = np.array(epsilon_low).flatten()
#     epsilon_hi = np.array(epsilon_hi).flatten()
            
#     # Construct PIs for test data
#     f_hat_t_batch = np.zeros((test_y.shape[0], test_y.shape[1], 3, P['B']))
#     for b, model_b in enumerate(ensemble_models):
#         f_hat_t_batch[:,:,:,b] = model_b.transform(test_x)
#     PI  = np.mean(f_hat_t_batch,  axis=-1) 
    
#     # Conformalize prediction intervals on the test data
#     conf_PI = np.zeros((test_y.shape[0], test_y.shape[1], 3))
#     conf_PI[:,:,1] = PI[:,:,1]
#     for i in range(test_y.shape[0]):   
#         e_quantile_lo = np.quantile(epsilon_low, 1-P['alpha']/2)
#         e_quantile_hi = np.quantile(epsilon_hi, 1-P['alpha']/2)
#         conf_PI[i,:,0] = PI[i,:,0] - e_quantile_lo
#         conf_PI[i,:,2] = PI[i,:,2] + e_quantile_hi
    
#         # update epsilon with the last s steps
#         e_lo, e_hi = utils.asym_nonconformity(label=test_y[i,:],
#                                                 low=PI[i,:,0],
#                                                 high=PI[i,:,2])
#         epsilon_low = np.delete(epsilon_low,slice(0,s,1))
#         epsilon_hi = np.delete(epsilon_hi,slice(0,s,1))
#         epsilon_low = np.append(epsilon_low, e_lo)
#         epsilon_hi = np.append(epsilon_hi, e_hi)
        
#     return PI, conf_PI


    # # Make training batches
    # batch_len = int(np.floor(train_x.shape[0]/B))
    # to_del = time_steps_in//time_steps_out # make sure there are no overlapping windows across batches
    # train_data = []
    # for b in range(B):
    #     train_data.append([train_x[b*batch_len:(b+1)*batch_len-to_del],\
    #          train_y[b*batch_len:(b+1)*batch_len-to_del]])