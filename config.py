M          = 3  #number of baseline models, which is M in paper.
max_epochs = 500
l_rate     = 1e-4
activation = 'tanh'  #nn.Tanh,              nn.ReLU, nn.Sigmoid
batch_size = 512
dropout    = 0.2
replace    = False
symmetric  = True
saveflag   = True
# save_dir   = './results/'
step       = 2
device     = 'cuda'
verbose    = True

alpha_set = np.array([0.05, 0.10, 0.15])
num_alpha = len(alpha_set)
alpha_base = max(alpha_set)
quantiles = [max(alpha_set)/2, 1 - max(alpha_set)/2]
# quantiles = np.zeros(2*num_alpha)
# for i in range(num_alpha):
#     quantiles[i] = alpha_set[i] / 2
#     quantiles[-(i+1)] = 1 - alpha_set[i] / 2

# loss_fn = PinballLoss(quantiles=quantiles, device=device)
input_dim = X_train.shape[1]
x_size = len(df.columns)
out_dim = len(quantiles)
kernel_size = 2
num_repeat = 1
hidden_units = [20 + i*4 for i in range(num_repeat)]
channel_sizes = [3 + i*2 for i in range(num_repeat)]