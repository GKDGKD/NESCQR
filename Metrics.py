import numpy as np
import os
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import pandas as pd
import matplotlib.pyplot as plt

def point_indexs(Y_preds, Y_test):
    MSE = mean_squared_error(Y_preds, Y_test)
    MAE = mean_absolute_error(Y_preds, Y_test)
    RMSE = np.sqrt(MSE)
    print('Test MAE:', MAE)
    print('Test RMSE:', RMSE)
    
    return MSE, MAE, RMSE

def interval_indexs(y_test, y_lower, y_upper):
    """ Compute average coverage and length of prediction intervals

    Parameters
    ----------

    y_test : numpy array, true labels (n)
    y_lower : numpy array, estimated lower bound for the labels (n)
    y_upper : numpy array, estimated upper bound for the labels (n)

    Returns
    -------

    coverage : float, average coverage
    avg_length : float, average length+

    """
    in_the_range = np.sum((y_test >= y_lower) & (y_test <= y_upper))
    coverage = in_the_range / len(y_test) * 100
    avg_length = np.mean(abs(y_upper - y_lower))
    print("PICP:",coverage)
    print("MPIW:", avg_length)
    
    return coverage, avg_length


def crps(fcst, obs):
    # fcst:预测值, ndarray
    # obs:观测值，即真实值
    fcst, obs = np.array(fcst), np.array(obs)
    if fcst.ndim < 2:
        fcst = fcst[:, np.newaxis]
        
    if obs.ndim < 2:
        obs = obs[:, np.newaxis]
        
    m = fcst.shape[1]
    crps_r = np.sum(abs(fcst-obs),axis=1)/m
    crps_l = 0
    for i in range(m):
        for j in range(m):
            crps_l = crps_l+abs(fcst[:,i]-fcst[:,j])
    crps_l = crps_l/(2*m*m)
    crps_result = np.mean(crps_r-crps_l)
    # print("CRPS:",crps_result)
    
    return crps_result
  
def sde(Y_preds, Y_test):
    '''
    类似于方差，越小预测越集中。
    '''
    
    errors = Y_test-Y_preds
    er_mu = np.mean(errors)   
    SDE = np.sqrt(np.mean((errors-er_mu)**2))
#     print("SDE: ",SDE)
    
    return SDE

def cross_loss(fcst):
    m = fcst.shape[1]
    n = fcst.shape[0]
    loss_1 = 0
    for i in range(m-1):
        for j in range(n):
            loss = max(0,fcst[j,i]-fcst[j,i+1])
            loss_1 += loss
#     print("Cross_loss:",loss_1/n)
    
    return loss_1/n


def reliability(y_pred, y_true, alpha):
    '''
    思想类似于MAE，如果预测值和实际值越接近，那么reliability也越接近于0.
    '''
    
    assert y_pred.shape[0] == y_true.shape[0]
    n = y_pred.shape[0]
    b = np.sum(y_pred > y_true)/n
    result = alpha - b
#     print("Reliablity at",alpha,"is:",result)
    
    return result

def interval_score(y_test, lower, upper, alpha):
    """
    PICP和MPIW的综合，越小越准确。The combination of PICP and MPIW, smaller is better.
    
    Reference:
    Gneiting, Tilmann, and Adrian E. Raftery. "Strictly proper scoring rules, prediction, and estimation." Journal of the American statistical Association 102.477 (2007): 359-378.
    """
    
    score = upper - lower + 2 / alpha * (lower - y_test) * (y_test < lower) + \
                            2 / alpha * (y_test - upper) * (y_test > upper)
    return np.mean(score)

def metrics(y_test, y_lower, y_upper, alpha, ita=0.5, saveflag=False, save_dir=None):
    '''
    回归 + 区间预测评估(单个区间)。
    
    input:
    y_pred: 回归结果，区间预测的话以上下界的中点为回归结果。注意所有y都为单列向量。
    y_test: 测试集的y
    y_lower: 预测区间下界
    y_upper: 预测区间上界
    alpha: 预测区间的置信水平，即miscoverage rate.
    ita: CWC中的一个参数，控制惩罚项的大小，注意这里的PICP和PINC均乘了100，所以这个参数不需要太大。
    
    output:
    res: 包括回归和区间的评估的结果
    '''
    y_test = np.array(y_test)
    if y_test.ndim > 1:
        y_test = y_test.squeeze()
    y_lower, y_upper = np.array(y_lower), np.array(y_upper)
    y_pred = (y_lower + y_upper) / 2
    assert y_pred.shape == y_test.shape == y_lower.shape == y_upper.shape
    PINC = (1 - alpha) * 100
    
    # Regression
    MSE = mean_squared_error(y_pred, y_test)
    MAE = mean_absolute_error(y_pred, y_test)
    RMSE = np.sqrt(MSE)
    CRPS = crps(y_pred, y_test)
    SDE = sde(y_pred, y_test)
    print('Regression: {:.4f}, MSE: {:.4f}, RMSE: {:.4f}, CRPS: {:.4f}, SDE: {:.4f}'.format(PINC, MAE,
                                    MSE, RMSE, CRPS, SDE))
    
    # Presdicion interval
    in_the_range = np.sum((y_test >= y_lower) & (y_test <= y_upper))
    PICP = in_the_range / len(y_test) * 100
    MPIW  = np.mean(abs(y_upper - y_lower))
    PINAW = MPIW / (y_test.max() - y_test.min())
    F = 2 * PICP * (1/MPIW) / (PICP + 1 / MPIW)
    ACE = PICP - PINC
    gamma = 1 if ACE < 0 else 0
    CWC = PINAW * (1 + gamma * np.exp(-ita*ACE))
    score = interval_score(y_test, y_lower, y_upper, alpha)

    print('PINC: {:.0f}%, PICP: {:.4f}, MPIW: {:.4f}, PINAW: {:.4f}, F: {:.4f}, ACE: {:.4f}, CWC: {:.4f}, interval_score: {:.4f}'.format(PINC, PICP,
                                    MPIW, PINAW, F, ACE, CWC, score))

    res = {
        'MAE': MAE,
        'MSE': MSE,
        'RMSE': RMSE,
        'CRPS': CRPS,
        'SDE': SDE,
        'PICP': PICP,
        'MPIW': MPIW,
        'F': F,
        'PINAW': PINAW,
        'ACE': ACE,
        'CWC': CWC,
        'Interval score': score
    }
    df = pd.DataFrame(res)
    if saveflag:
        df.to_csv(os.path.join(save_dir, 'metrics.csv'))
        print('Metrics are saved to {}'.format(os.path.join(save_dir, 'metrics.csv')))
        return df
    else:
        return df

def evaluate(y_test, PIs, alpha_set, ita=0.5, saveflag=False, save_dir=None, verbose=True):
    """
    对所有预测区间进行评估。

    Args:
    y_test: 测试集的y, the ground truth.
    PIs: 预测区间，从左到右依次增大，prediction intervals, ndarray.
    alpha_set: 置信水平集合, the set of alphas.
    ita: CWC中的一个参数，控制惩罚项的大小，注意这里的PICP和PINC均乘了100，所以这个参数不需要太大。
    verbose: 是否输出详细信息。
    
    out:
    df: 包括回归和区间的评估的结果, DataFrame
    
    """
    assert PIs.shape[1] == len(alpha_set) * 2
    y_test = np.array(y_test)
    if y_test.ndim > 1:
        y_test = y_test.squeeze()
    
    result = []
    for i, alpha in enumerate(alpha_set):
        y_lower, y_upper = PIs[:, i], PIs[:, -(i+1)]
        y_pred = (y_lower + y_upper) / 2   # Regression prediction
        PINC = (1 - alpha) * 100
    
        # Regression
        MSE = mean_squared_error(y_pred, y_test)
        MAE = mean_absolute_error(y_pred, y_test)
        RMSE = np.sqrt(MSE)
        CRPS = crps(y_pred, y_test)
        SDE = sde(y_pred, y_test)

        # Interval prediction
        in_the_range = np.sum((y_test >= y_lower) & (y_test <= y_upper))
        PICP = in_the_range / len(y_test) * 100
        MPIW  = np.mean(abs(y_upper - y_lower))
        PINAW = MPIW / (y_test.max() - y_test.min())
        F = 2 * PICP * (1/MPIW) / (PICP + 1 / MPIW)
        ACE = PICP - PINC
        gamma = 1 if ACE < 0 else 0
        CWC = PINAW * (1 + gamma * np.exp(-ita*ACE))
        score = interval_score(y_test, y_lower, y_upper, alpha)

        if verbose:
            print('PINC: {:.0f}%, MAE: {:.4f}, MSE: {:.4f}, RMSE: {:.4f}, CRPS: {:.4f}, SDE: {:.4f}'.format(PINC, MAE,
                                            MSE, RMSE, CRPS, SDE))
            print('PINC: {:.0f}%, PICP: {:.4f}, MPIW: {:.4f}, PINAW: {:.4f}, F: {:.4f}, ACE: {:.4f}, CWC: {:.4f}, \
                  interval_score: {:.4f}'.format(PINC, PICP, MPIW, PINAW, F, ACE, CWC, score))
        result.append([PINC, MAE, MSE, RMSE, CRPS, SDE, PICP, MPIW, F, PINAW, ACE, CWC, score])

    result_df = pd.DataFrame(result, columns=['PINC','MAE', 'MSE', 'RMSE', 'CRPS', 'SDE', 'PICP', 'MPIW', 'F', 'PINAW', 'ACE', 'CWC', 'Interval score'])
    if saveflag:
        result_df.to_csv(os.path.join(save_dir, 'metrics.csv'))
        print('Metrics are saved to {}'.format(os.path.join(save_dir, 'metrics.csv')))
        return result_df
    else:
        return result_df


def cross_bound_check(prediction, saveflag=False, save_dir=None):
    '''
    检查是否有区间耦合、交叉现象。
    
    input:
    prediction: [m, n], prediction intervals, 不同置信水平下的预测区间，如[0.05,0.075, 0.925, 0.95];
    m 为样本个数，n 为区间个数。
    注意prediction每一列从左到右，是逐渐增大的区间。
    
    output:
    l: [m, n/2 - 1], 存储下界是否有交叉的结果。
    u: [m, n/2 - 1], 存储上界是否有交叉的结果。
    
    '''
    
    assert prediction.shape[1] > 1
    
    prediction = np.array(prediction)
    m, n = prediction.shape
    num_PI = int(n/2)

    PI_lower = prediction[:, 0:num_PI]
    PI_upper = prediction[:, num_PI:]

    l, u = np.zeros((m, num_PI-1)), np.zeros((m, num_PI-1))
    MUCW, MLCW = 0, 0

    for r in range(num_PI-1):
        l[:, r] = PI_lower[:, r] > PI_lower[:, r+1]
        u[:, r] = PI_upper[:, r] > PI_upper[:, r+1]
        m1 = u[:, r] * (PI_upper[:, r] - PI_upper[:, r+1])
        u1 = u[:, r].sum()
        m2 = l[:, r] * (PI_lower[:, r] - PI_lower[:, r+1])
        l1 = l[:, r].sum()

        print('r = %d, l1 = %.4f, u1 = %.4f'%(r, l1, u1))
        if u1 == 0:
            MUCW += 0
        else:
            MUCW += m1.sum() / u1

        if l1 == 0:
            MLCW += 0
        else:
            MLCW += m2.sum() / l1
            
    if u.sum() or l.sum() > 0:
        print('Cross-bound phenomenon exists.')
        print('l.sum() = ', l.sum())
        print('u.sum() = ', u.sum())
        print('MUCW = %.4f, MLCW = %.4f'%(MUCW, MLCW))
    else:
        print('No cross-bound phenomenon.')
        print('MUCW = %.4f, MLCW = %.4f'% (MUCW, MLCW))
        
    Cross_loss = cross_loss(prediction)
    print('Cross loss: ', Cross_loss)

    df = pd.DataFrame({'MUCW': [MUCW], 'MLCW': [MLCW], 'Cross loss': [Cross_loss]})
    if saveflag:
        df.to_csv(os.path.join(save_dir, 'metrics_cross.csv'))
        print('Metrics for quantile crossing are saved to {}'.format(os.path.join(save_dir, 'metrics_cross.csv')))
        return df
    else:
        return df