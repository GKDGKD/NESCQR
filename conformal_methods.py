import numpy as np

def CQRS(res_test, res_cal, y_cal, alpha_set:list):
    '''
    Conformalized quantile regression using symmetric conformity scores.
    
    input:
    res_test: the prediction intervals on the test set, torch.tensor or nd.array, [n, num_alpha*2].
    res_cal: the prediction intervals on the test set, torch.tensor or nd.array, [n, num_alpha*2].
    y_cal: the ground truths of y on the test set, torch.tensor or nd.array, [n,1].
    alpha_set: the list of confidence levels, from small to large, such as [0.05, 0.10, 0.15].
    
    output:
    C: the precition intervals after conformalization, nd.array, [n, num_alpha*2].
    Note that the columns in the prediction intervals are from small to large.
    C[:, i] and C[:, -(i+1)] form the predicition interval at confidence level alpha_set[i].
    '''
    
    res_test, res_cal, y_cal, alpha_set = np.array(res_test), np.array(res_cal), np.array(y_cal), np.array(alpha_set)
    assert res_cal.shape[0] == y_cal.shape[0]

    if y_cal.ndim < 1:
        y_cal = y_cal[:, np.newaxis]
        
    cal_size = y_cal.shape[0]
    test_size = res_test.shape[0]
    num_alpha = len(alpha_set)

    C = np.zeros((test_size, num_alpha*2)) #Initialize the prediction intervals.
    Q = np.zeros(num_alpha) 
    E = np.zeros((cal_size, num_alpha))  #conformity scores
    
    for i, alpha in enumerate(alpha_set):
        E[:, i] = np.max((res_cal[:,i] - y_cal[:,0], y_cal[:, 0] - res_cal[:,-(i+1)]), axis=0)
        Q[i] = np.quantile(E[:, i], (1-alpha)*(1+1/cal_size))
        C[:, i] = res_test[:, i] - Q[i]
        C[:, -(i+1)] = res_test[:, -(i+1)] + Q[i]
        
    return  C


def CQRA(res_test, res_cal, y_cal, alpha_set):
    '''
    Conformalized quantile regression using asymmetric conformity scores.
    
    input:
    res_test: the prediction intervals on the test set, torch.tensor or nd.array, [n, num_alpha*2].
    res_cal: the prediction intervals on the test set, torch.tensor or nd.array, [n, num_alpha*2].
    y_cal: the ground truths of y on the test set, torch.tensor or nd.array, [n,1].
    alpha_set: the list of confidence levels, from small to large, such as [0.05, 0.10, 0.15].
    
    output:
    C: the precition intervals after conformalization, nd.array, [n, num_alpha*2].
    Note that the columns in the prediction intervals are from small to large.
    C[:, i] and C[:, -(i+1)] form the predicition interval at confidence level alpha_set[i].
    '''
    
    res_test, res_cal, y_cal, alpha_set = np.array(res_test), np.array(res_cal), np.array(y_cal), np.array(alpha_set)
    assert res_cal.shape[0] == y_cal.shape[0]
    
    if y_cal.ndim < 1:
        y_cal = y_cal[:, np.newaxis]
        
    cal_size = y_cal.shape[0]
    test_size = res_test.shape[0]
    num_alpha = len(alpha_set)
    
    C = np.zeros((test_size, num_alpha*2))
    Q_low, Q_high = np.zeros(num_alpha), np.zeros(num_alpha)
    E_low, E_high = np.zeros((cal_size, num_alpha)), np.zeros((cal_size, num_alpha))
    for i in range(num_alpha):
        E_low[:, i] = res_cal[:, i] - y_cal[:, 0]
        E_high[:, i] = y_cal[:, 0] - res_cal[:, i+num_alpha]
    
    for i in range(num_alpha):
        alpha_low = alpha_set[i] / 2 #Note that alpha = alpha_low + alpha_high, where alpha_low = alpha_high = alpha / 2
        # alpha_high = alpha_low
        Q_low[i] = np.quantile(E_low[:,i], 1 - alpha_low)
        Q_high[-(i+1)] = np.quantile(E_high[:,-(i+1)], 1 - alpha_low)

        C[:, i] = res_test[:, i] - Q_low[i]
        C[:, -(i+1)] = res_test[:, -(i+1)] + Q_high[-(i+1)]

    return  C

def DCQRS(res_test, y_test, res_cal, y_cal, alpha_set:list, step:int):
    '''
    Dynamic Conformalized quantile regression using symmetric conformity scores.
    
    input:
    res_test: the prediction intervals on the test set, torch.tensor or nd.array, [n, num_alpha*2].
    y_test: the ground truths of y on the test set, torch.tensor or nd.array, [n,1].
    res_cal: the prediction intervals on the test set, torch.tensor or nd.array, [n, num_alpha*2].
    y_cal: the ground truths of y on the test set, torch.tensor or nd.array, [n,1].
    alpha_set: the list of confidence levels, from small to large, such as [0.05, 0.10, 0.15].
    step: an integer to control the update speed of the lists of conformity scores, a small integer is recommended, such as 2.
    
    output:
    C: the precition intervals after conformalization, nd.array, [n, num_alpha*2].
    Note that the columns in the prediction intervals are from small to large.
    C[:, i] and C[:, -(i+1)] form the predicition interval at confidence level alpha_set[i].
    '''
    
    res_test, res_cal, y_cal, alpha_set = np.array(res_test), np.array(res_cal), np.array(y_cal), np.array(alpha_set)
    y_test = np.array(y_test)
    assert res_cal.shape[0] == y_cal.shape[0]
    assert res_test.shape[0] == y_test.shape[0]
    if y_cal.ndim < 1:
        y_cal = y_cal[:, np.newaxis]
        
    cal_size = y_cal.shape[0]
    test_size = y_test.shape[0]
    y_all = np.concatenate((y_cal, y_test), axis=0)
    res_all = np.concatenate((res_cal, res_test), axis=0)
    num_alpha = len(alpha_set)
       
    C = np.zeros((test_size, num_alpha*2))
    Q = np.zeros(num_alpha)
    E = np.zeros((cal_size, num_alpha))
    
    for i, alpha in enumerate(alpha_set):
        E[:, i] = np.max((res_cal[:,i] - y_cal[:,0], y_cal[:, 0] - res_cal[:,-(i+1)]), axis=0)

    for t in range(cal_size, cal_size+test_size):
        for i, alpha in enumerate(alpha_set):
            Q[i] = np.quantile(E[:,i], (1-alpha)*(1+1/cal_size))
            C[t-cal_size, i] = res_all[t, i] - Q[i]
            C[t-cal_size, -(i+1)] = res_all[t, -(i+1)] + Q[i]

        # Update the lists of conformity scores
        if t % step == 0:
    #         print('t = %d, Q[0] = %f, E.shape = %s, E[:,0].mean() = %f.' % (t, Q[0], str(E_high.shape), E[:,0].mean()))
            for j in range(t - step, t - 1):
                for i in range(num_alpha):
                    e = np.max((res_all[j, i] - y_all[j], y_all[j] - res_all[j, -(i+1)]),axis=0)  #这里可能有问题
                    E_temp = np.delete(E[:, i], 0, 0)
                    E_temp = np.append(E_temp, e)
                    E[:, i] = E_temp   
        
    return  C

def DCQRA(res_test, y_test, res_cal, y_cal, alpha_set, step):
    '''
    Dynamic Conformalized quantile regression using asymmetric conformity scores.
    
    input:
    res_test: the prediction intervals on the test set, torch.tensor or nd.array, [n, num_alpha*2].
    y_test: the ground truths of y on the test set, torch.tensor or nd.array, [n,1].
    res_cal: the prediction intervals on the test set, torch.tensor or nd.array, [n, num_alpha*2].
    y_cal: the ground truths of y on the test set, torch.tensor or nd.array, [n,1].
    alpha_set: the list of confidence levels, from small to large, such as [0.05, 0.10, 0.15].
    step: an integer to control the update speed of the lists of conformity scores, a small integer is recommended, such as 2.
    
    output:
    C: the precition intervals after conformalization, nd.array, [n, num_alpha*2].
    Note that the columns in the prediction intervals are from small to large.
    C[:, i] and C[:, -(i+1)] form the predicition interval at confidence level alpha_set[i].
    '''
    
    res_test, res_cal, y_cal, alpha_set = np.array(res_test), np.array(res_cal), np.array(y_cal), np.array(alpha_set)
    y_test = np.array(y_test)
    assert res_cal.shape[0] == y_cal.shape[0]
    assert res_test.shape[0] == y_test.shape[0]
    if y_cal.ndim < 1:
        y_cal = y_cal[:, np.newaxis]
        
    cal_size = y_cal.shape[0]
    test_size = y_test.shape[0]
    y_all = np.concatenate((y_cal, y_test), axis=0)
    res_all = np.concatenate((res_cal, res_test), axis=0)
    num_alpha = len(alpha_set)

    # Compute the conformity score E.
    C = np.zeros((test_size, num_alpha*2))
    Q_low, Q_high = np.zeros(num_alpha), np.zeros(num_alpha)
    E_low, E_high = np.zeros((cal_size, num_alpha)), np.zeros((cal_size, num_alpha))
    for i in range(num_alpha):
        E_low[:, i] = res_cal[:, i] - y_cal[:, 0]
        E_high[:, i] = y_cal[:, 0] - res_cal[:, i+num_alpha]
    
    # Conformalize
    for t in range(cal_size, cal_size+test_size):
        for i in range(num_alpha):
            alpha_low = alpha_set[i] / 2
            # alpha_high = alpha_low
            Q_low[i] = np.quantile(E_low[:,i], (1-alpha_low))
            Q_high[-(i+1)] = np.quantile(E_high[:,-(i+1)], (1-alpha_low))

            C[t-cal_size, i] = res_all[t, i] - Q_low[i]
            C[t-cal_size, -(i+1)] = res_all[t, -(i+1)] + Q_high[-(i+1)]

        # Update the lists of conformity scores
        if t % step == 0:
#             print('t = %d, Q_low[0] = %f, Q_high[-1] = %f, E_low.shape = %s, E_high.shape = %s.' % 
#               (t,Q_low[0],Q_high[-1],str(E_low.shape), str(E_high.shape)))
            for j in range(t-step, t-1):
                for i in range(num_alpha):
                    e_low = res_all[j, i] - y_all[j]
                    e_high = y_all[j] - res_all[j, -(i+1)]
                    E_low_temp = np.delete(E_low[:,i], 0, 0)    #删除第一个元素
                    E_low_temp = np.append(E_low_temp, e_low)   #添加新的元素
                    E_low[:,i] = E_low_temp
                    E_high_temp = np.delete(E_high[:,-(i+1)], 0, 0)
                    E_high_temp = np.append(E_high_temp, e_high)
                    E_high[:,-(i+1)] = E_high_temp              
        
    return  C

def MCQRS(res_cal, y_cal, res_test, alpha_set):
    '''
    Montone conformalized quantile regression using symmetric conformity scores.
    
    input:
    res_test: the prediction interval on the test set, torch.tensor or nd.array, [n, 2].
    res_cal: the prediction interval on the test set, torch.tensor or nd.array, [n, 2].
    y_cal: the ground truths of y on the test set, torch.tensor or nd.array, [n,1].
    alpha_set: the list of confidence levels, from small to large, such as [0.05, 0.10, 0.15].
    
    output:
    C: the precition intervals after conformalization, nd.array, [n, num_alpha*2].
    Note that the columns in the prediction intervals are from small to large.
    C[:, i] and C[:, -(i+1)] form the predicition interval at confidence level alpha_set[i].
    '''

    res_cal, res_test, y_cal = np.array(res_cal), np.array(res_test), np.array(y_cal)
    assert res_cal.shape[0] == y_cal.shape[0]
    assert res_cal.shape[1] == 2
    assert res_test.shape[1] == 2

    test_size = res_test.shape[0]
    num_alpha = len(alpha_set)
    C = np.zeros((test_size, num_alpha*2))
    
    # Compute the conformity score E.
    Q = np.zeros(num_alpha)
    E = np.max((res_cal[:, 0] - y_cal[:,0], y_cal[:,0] - res_cal[:, -1]), axis=0)

    # Conformalize
    for i, alpha in enumerate(alpha_set):
        Q[i] = np.quantile(E, (1-alpha)*(1+1/len(E)))
        C[:, i] = res_test[:, 0] - Q[i]
        C[:, -(i+1)] = res_test[:, -1] + Q[i]
        
    return  C

def MCQRA(res_cal, y_cal, res_test, alpha_set):
    '''
    Montone conformalized quantile regression using asymmetric conformity scores.
    
    input:
    res_test: the prediction interval on the test set, torch.tensor or nd.array, [n, 2].
    res_cal: the prediction interval on the test set, torch.tensor or nd.array, [n, 2].
    y_cal: the ground truths of y on the test set, torch.tensor or nd.array, [n,1].
    alpha_set: the list of confidence levels, from small to large, such as [0.05, 0.10, 0.15].
    
    output:
    C: the precition intervals after conformalization, nd.array, [n, num_alpha*2].
    Note that the columns in the prediction intervals are from small to large.
    C[:, i] and C[:, -(i+1)] form the predicition interval at confidence level alpha_set[i].
    '''

    res_cal, res_test, y_cal = np.array(res_cal), np.array(res_test), np.array(y_cal)
    assert res_cal.shape[0] == y_cal.shape[0]
    assert res_cal.shape[1] == 2
    assert res_test.shape[1] == 2

    test_size = res_test.shape[0]
    num_alpha = len(alpha_set)
    C = np.zeros((test_size, num_alpha*2))
    
    # Compute conformity score E.
    Q_low, Q_high = np.zeros(num_alpha), np.zeros(num_alpha)
    E_low = res_cal[:, 0] - y_cal[:,0]
    E_high = y_cal[:, 0] - res_cal[:, 1]
    
    # Conformalize
    for i, alpha in enumerate(alpha_set):
        q_low = alpha / 2
        q_high = 1 - q_low
        Q_low[i] = np.quantile(E_low, (1-q_low))
        Q_high[-(i+1)] = np.quantile(E_high, q_high)

        C[:, i] = res_test[:, 0] - Q_low[i]
        C[:, -(i+1)] = res_test[:, -1] + Q_high[-(i+1)]

    return  C

def DMCQRS(res_cal, y_cal, res_test, Y_test, alpha_set, step):
    '''
    Dynamic Monotone conformalized quantile regression using symmetric conformity scores.
    
    input:
    res_test: the prediction intervals on the test set, torch.tensor or nd.array, [n, num_alpha*2].
    y_test: the ground truths of y on the test set, torch.tensor or nd.array, [n,1].
    res_cal: the prediction intervals on the test set, torch.tensor or nd.array, [n, num_alpha*2].
    y_cal: the ground truths of y on the test set, torch.tensor or nd.array, [n,1].
    alpha_set: the list of confidence levels, from small to large, such as [0.05, 0.10, 0.15].
    step: an integer to control the update speed of the lists of conformity scores, a small integer is recommended, such as 2.
    
    output:
    C: the precition intervals after conformalization, nd.array, [n, num_alpha*2].
    Note that the columns in the prediction intervals are from small to large.
    C[:, i] and C[:, -(i+1)] form the predicition interval at confidence level alpha_set[i].
    '''

    res_cal, res_test, y_cal = np.array(res_cal), np.array(res_test), np.array(y_cal)
    Y_test = np.array(Y_test)
    assert res_cal.shape[0] == y_cal.shape[0]
    assert res_cal.shape[1] == 2
    assert res_test.shape[1] == 2

    low = res_cal[:, 0]
    up = res_cal[:, -1]
    num_alpha = len(alpha_set)

    low, up, y_cal = np.array(low), np.array(up), np.array(y_cal)
    y_all = np.concatenate((y_cal, Y_test), axis=0)
    res_all = np.concatenate((res_cal, res_test), axis=0)

    test_size = Y_test.shape[0]
    C = np.zeros((test_size, num_alpha*2))
    Q = np.zeros(num_alpha)

    E = np.max((low - y_cal[:,0], y_cal[:,0] - up), axis=0)
    cal_size = len(E)

    for t in range(cal_size, cal_size+test_size):
        for i, alpha in enumerate(alpha_set):
            Q[i] = np.quantile(E, (1-alpha)*(1+1/len(E)))
            C[:, i] = res_test[:, 0] - Q[i]
            C[:, -(i+1)] = res_test[:, -1] + Q[i]

        if t % step == 0:
    #         print('t = %d, Q[0] = %f, E.shape = %s, E.mean() = %f.' % (t, Q[0], str(E.shape), E.mean()))
            for j in range(t - step, t - 1):
                e = np.max((res_all[j, 0] - y_all[j,0], y_all[j,0] - res_all[j, -1]),axis=0)
                E_temp = np.delete(E, 0, 0)
                E_temp = np.append(E_temp, e)
                E = E_temp       
        
    return  C

def DMCQRA(res_cal, y_cal, res_test, Y_test, alpha_set, step):
    '''
    Dynamic Monotone conformalized quantile regression using asymmetric conformity scores.
    
    input:
    res_test: the prediction intervals on the test set, torch.tensor or nd.array, [n, num_alpha*2].
    y_test: the ground truths of y on the test set, torch.tensor or nd.array, [n,1].
    res_cal: the prediction intervals on the test set, torch.tensor or nd.array, [n, num_alpha*2].
    y_cal: the ground truths of y on the test set, torch.tensor or nd.array, [n,1].
    alpha_set: the list of confidence levels, from small to large, such as [0.05, 0.10, 0.15].
    step: an integer to control the update speed of the lists of conformity scores, a small integer is recommended, such as 2.
    
    output:
    C: the precition intervals after conformalization, nd.array, [n, num_alpha*2].
    Note that the columns in the prediction intervals are from small to large.
    C[:, i] and C[:, -(i+1)] form the predicition interval at confidence level alpha_set[i].
    '''

    res_cal, res_test, y_cal = np.array(res_cal), np.array(res_test), np.array(y_cal)
    Y_test = np.array(Y_test)
    assert res_cal.shape[0] == y_cal.shape[0]
    assert res_cal.shape[1] == 2
    assert res_test.shape[1] == 2

    low = res_cal[:, 0]
    up = res_cal[:, -1]
    num_alpha = len(alpha_set)
    test_size = res_test.shape[0]
    cal_size = y_cal.shape[0]

    y_all = np.concatenate((y_cal, Y_test), axis=0)
    res_all = np.concatenate((res_cal, res_test), axis=0)
    C = np.zeros((test_size, num_alpha*2))
    
    # Compute conformity score E. 
    Q_low, Q_high = np.zeros(num_alpha), np.zeros(num_alpha)
    E_low = low - y_cal[:,0]
    E_high = y_cal[:, 0] - up
        
    for t in range(cal_size, cal_size+test_size):
        for i, alpha in enumerate(alpha_set):
            q_low = alpha / 2
            q_high = 1 - q_low
            Q_low[i] = np.quantile(E_low, (1-q_low))
            Q_high[-(i+1)] = np.quantile(E_high, q_high)
            
            C[:, i] = res_test[:, 0] - Q_low[i]
            C[:, -(i+1)] = res_test[:, -1] + Q_high[-(i+1)]

        if t % step == 0:
    #         print('t = %d, Q[0] = %f, E.shape = %s, E.mean() = %f.' % (t, Q[0], str(E.shape), E.mean()))
            for j in range(t - step, t - 1):
                e_low = res_all[j, 0] - y_all[j]
                e_high = y_all[j] - res_all[j, -1]
                E_low_temp = np.delete(E_low, 0, 0)    #remove the first element
                E_low_temp = np.append(E_low_temp, e_low)   #add new element 
                E_low = E_low_temp
                E_high_temp = np.delete(E_high, 0, 0)
                E_high_temp = np.append(E_high_temp, e_high)
                E_high = E_high_temp  

    return C

