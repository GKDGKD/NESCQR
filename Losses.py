import torch
import numpy as np
import copy
import torch.nn as nn

class PinballLoss(nn.Module):
    """ Pinball loss at all confidence levels.
    """
    def __init__(self, quantiles, device='cpu'):
        """ Initialize

        Parameters
        ----------
        quantiles : the list of quantiels, [1, n], list or ndarray. 
        For example, quantiles =  [0.025 0.05  0.075 0.925 0.95  0.975].

        """
        super(PinballLoss, self).__init__()
        self.device = device
        # self.quantiles = quantiles
        self.quantiles = torch.tensor(quantiles, device=self.device)

    def forward(self, preds, target):
        """ Compute the pinball loss

        Parameters
        ----------
        preds : the predictions at all quantiles, [n, len(quantiles)], torch.tensor.
        target : the ground truth of y, [n, 1], torch.tensor.

        Returns
        -------
        loss : the value of the pinball loss.

        """

        assert preds.shape[0] == target.shape[0]

        # if target.ndim == 1:
        #     target = target.unsqueeze(dim=1)

        # losses = []

        # for i, q in enumerate(self.quantiles):
        #     errors = target[:, 0] - preds[:, i]
        #     losses.append(torch.max((q - 1) * errors, q * errors).unsqueeze(1))\

        # losses = []

        # for i, q in enumerate(self.quantiles):
        #     print(f'target.device: {target.device}, preds[:, i].device: {preds[:, i].device}')
        #     errors = target - preds[:, i]
        #     losses.append(torch.max((q - 1) * errors, q * errors).unsqueeze(1))

        # loss = torch.mean(torch.sum(torch.cat(losses, dim=1), dim=1))

        # 计算分位数损失函数
        losses = torch.zeros_like(self.quantiles, device=target.device)

        for i, q in enumerate(self.quantiles):
            # 显式将 pred 移动到 GPU 上，确保与 target 在同一设备上
            pred_i = preds[:, i].to(target.device)
            
            errors = target - pred_i
            losses[i] = torch.mean(torch.max((q - 1) * errors, q * errors))

        # 计算总损失
        loss = torch.mean(losses)

        return loss


class pinball_l1_loss(nn.Module):
    # pinball loss + L1 regularization
    def __init__(self, quantiles, model, Lambda=1.0):
        super().__init__()
        self.quantiles = quantiles  # pytorch vector of quantile levels, each in the range (0,1)
        self.Lambda = Lambda  # the coefficent of penalty, float
        self.model = model

    def forward(self, preds, target):
        """ Compute the pinball loss with L1 regularization

        Parameters
        ----------
        model: nn.module, like NET model above
        preds : pytorch tensor of estimated labels (n)
        truth : pytorch tensor of true labels (n)

        Returns
        -------
        loss : cost function value

        """
        assert not target.requires_grad
        assert preds.size(0) == target.size(0)
        losses = []

        for i, q in enumerate(self.quantiles):
            errors = target[:, 0] - preds[:, i]
            losses.append(torch.max((q - 1) * errors, q * errors).unsqueeze(1))

        loss_pinball = torch.mean(torch.sum(torch.cat(losses, dim=1), dim=1))

        l1 = torch.tensor(0.)
        for param in self.model.parameters():
            l1 += torch.sum(torch.abs(param))

        loss = loss_pinball + self.Lambda * l1

        return loss


class pinball_l2_loss(nn.Module):
    # pinball loss + L2 regularization
    def __init__(self, quantiles, model, Lambda=1.0):
        super().__init__()
        self.quantiles = quantiles  # pytorch vector of quantile levels, each in the range (0,1)
        self.Lambda = Lambda        # the coefficent of penalty, float
        self.model = model          # nn.module, like NET model above

    def forward(self, preds, target):
        """ Compute the pinball loss with L1 regularization

        Parameters
        ----------
        preds : pytorch tensor of estimated labels (n)
        truth : pytorch tensor of true labels (n)

        Returns
        -------
        loss : cost function value

        """
        assert not target.requires_grad
        assert preds.size(0) == target.size(0)
        losses = []

        for i, q in enumerate(self.quantiles):
            errors = target[:, 0] - preds[:, i]
            losses.append(torch.max((q - 1) * errors, q * errors).unsqueeze(1))

        loss_pinball = torch.mean(torch.sum(torch.cat(losses, dim=1), dim=1))

        l2 = torch.tensor(0.)
        for param in self.model.parameters():
            l2 += torch.norm(param)

        loss = loss_pinball + self.Lambda * l2

        return loss

    
class pinball_bayes_loss(nn.Module):
    # pinball loss + bayes regularization
    def __init__(self, model,  quantiles, hidden, W1_lambda, b1_lambda, W_last_lambda, W1_anc, b1_anc, W_last_anc,reg='anc', Lambda=1.0):
        super().__init__()
        self.model = model
        self.quantiles = quantiles  # pytorch vector of quantile levels, each in the range (0,1)
        self.reg = reg       #选择正则项
        
        self.W1_lambda = W1_lambda
        self.b1_lambda = b1_lambda
        self.W_last_lambda = W_last_lambda
        self.W1_anc = W1_anc
        self.b1_anc = b1_anc
        self.W_last_anc = W_last_anc
        
        self.hidden = hidden  #隐藏层神经元个数
        self.Lambda = Lambda  #惩罚项系数

    def forward(self, preds, target):
        """ Compute the pinball loss with L1 regularization

        Parameters
        ----------
        model: nn.module, like NET model above
        preds : pytorch tensor of estimated labels (n)
        target : pytorch tensor of true labels (n)

        Returns
        -------
        loss : cost function value

        """
#         target = torch.tensor(target)
#         preds = torch.tensor(preds)
        assert not target.requires_grad
        assert preds.size(0) == target.size(0)
        losses = []

        for i, q in enumerate(self.quantiles):
            errors = target - preds[:, i]
            losses.append(torch.max((q - 1) * errors, q * errors).unsqueeze(1))

        loss_pinball = torch.mean(torch.sum(torch.cat(losses, dim=1), dim=1))

        W1_anc = self.W1_anc
        b1_anc = self.b1_anc
        W_last_anc = self.W_last_anc
        
        n_data = preds.shape[0]

        # set up reg loss
        l2 = 0
        if self.reg == 'anc':
            l2 += self.W1_lambda / n_data * torch.mul(self.model[0].weight - W1_anc, self.model[0].weight - W1_anc).sum()
            l2 += self.b1_lambda / n_data * torch.mul(self.model[0].bias - b1_anc, self.model[0].bias - b1_anc).sum()
            l2 += self.W_last_lambda / n_data * torch.mul(self.model[2].weight - W_last_anc,
                                                     self.model[2].weight - W_last_anc).sum()
        elif self.reg == 'reg':
            l2 += self.W1_lambda / n_data * torch.mul(self.model[0].weight, self.model[0].weight).sum()
            l2 += self.b1_lambda / n_data * torch.mul(self.model[0].bias, self.model[0].bias).sum()
            l2 += self.W_last_lambda / n_data * torch.mul(self.model[2].weight, self.model[2].weight).sum()
        elif self.reg == 'free':
            # do nothing
            l2 += 0.0
        
        loss = loss_pinball + self.Lambda * l2

        return loss

class pinball_anchored_loss(nn.Module):
    # pinball loss + anchored regularization
    def __init__(self, model,  quantiles,  W_anc, W_lambda, reg='anc'):
        super().__init__()
        self.model = model
        self.quantiles = quantiles  # pytorch vector of quantile levels, each in the range (0,1)
        self.reg = reg       #regularizer: anc, reg, free
        
        self.W_anc = W_anc   # anchored term for each layer's weight and bias
        self.W_lambda = W_lambda    # weight of each parameters in regularizer
              

    def forward(self, preds, target):
        """ Compute the pinball loss with L1 regularization

        Parameters
        ----------
        model: nn.module, like NET model above
        preds : pytorch tensor of estimated labels (n)
        target : pytorch tensor of true labels (n)

        Returns
        -------
        loss : cost function value

        """
#         target = torch.tensor(target)
#         preds = torch.tensor(preds)
        assert not target.requires_grad
        assert preds.size(0) == target.size(0)
        losses = []

        #------------------------ pinball loss --------------------------------
        for i, q in enumerate(self.quantiles):
            errors = target[:, 0] - preds[:, i]
            losses.append(torch.max((q - 1) * errors, q * errors).unsqueeze(1))

        loss_pinball = torch.mean(torch.sum(torch.cat(losses, dim=1), dim=1))
        
        #------------------------ regularizer --------------------------------
        n_data = preds.shape[0]
        l2 = torch.tensor(0.)
        if self.reg == 'anc':
            for i, param in enumerate(self.model.parameters()):
                l2 += self.W_lambda[i] / n_data * torch.mul(param - torch.tensor(self.W_anc[i]), 
                                              param - torch.tensor(self.W_anc[i])).sum()
        elif self.reg == 'reg':
            # similar to L2 regularization
            for i, param in enumerate(self.model.parameters()):
                l2 += self.W_lambda[i] / n_data * torch.mul(param, param).sum()
        elif self.reg == 'free':
            # do nothing, no regularization
            l2 += 0
        
        loss = loss_pinball + l2

        return loss

class MQD_loss(nn.Module):
    # pinball loss + bayes regularization + MUCW + MLCW
    def __init__(self, model,  quantiles, hidden, W1_lambda, b1_lambda, W_last_lambda, W1_anc, b1_anc, W_last_anc,reg='anc', Lambda=1.0):
        super().__init__()
        self.model = model
        self.quantiles = quantiles  # pytorch vector of quantile levels, each in the range (0,1)
        self.reg = reg       #选择正则项
        
        self.W1_lambda = W1_lambda
        self.b1_lambda = b1_lambda
        self.W_last_lambda = W_last_lambda
        self.W1_anc = W1_anc
        self.b1_anc = b1_anc
        self.W_last_anc = W_last_anc
        
        self.hidden = hidden  #隐藏层神经元个数
        self.Lambda = Lambda  #惩罚项系数

    def forward(self, preds, target):
        """ Compute the pinball loss with L1 regularization

        Parameters
        ----------
        model: nn.module, like NET model above
        preds : pytorch tensor of estimated labels (n)
        target : pytorch tensor of true labels (n)

        Returns
        -------
        loss : cost function value

        """
#         target = torch.tensor(target)
#         preds = torch.tensor(preds)
        assert not target.requires_grad
        assert preds.shape[0] == target.shape[0]
        losses = []

        #----------------------- pinball loss -----------------------------------
        for i, q in enumerate(self.quantiles):
            errors = target[:, 0] - preds[:, i]
            losses.append(torch.max((q - 1) * errors, q * errors).unsqueeze(1))

        loss_pinball = torch.mean(torch.sum(torch.cat(losses, dim=1), dim=1))

        
        #---------------------- regularization ----------------------------------
        W1_anc = self.W1_anc
        b1_anc = self.b1_anc
        W_last_anc = self.W_last_anc
        n_data = preds.shape[0]
        
        l2 = 0
        if self.reg == 'anc':
            l2 += self.W1_lambda / n_data * torch.mul(self.model[0].weight - W1_anc, self.model[0].weight - W1_anc).sum()
            l2 += self.b1_lambda / n_data * torch.mul(self.model[0].bias - b1_anc, self.model[0].bias - b1_anc).sum()
            l2 += self.W_last_lambda / n_data * torch.mul(self.model[2].weight - W_last_anc,
                                                     self.model[2].weight - W_last_anc).sum()
        elif self.reg == 'reg':
            l2 += self.W1_lambda / n_data * torch.mul(self.model[0].weight, self.model[0].weight).sum()
            l2 += self.b1_lambda / n_data * torch.mul(self.model[0].bias, self.model[0].bias).sum()
            l2 += self.W_last_lambda / n_data * torch.mul(self.model[2].weight, self.model[2].weight).sum()
        elif self.reg == 'free':
            # do nothing
            l2 += 0.0
        
        #------------------- MUCW + MLCW ----------------------------------
        prediction = preds.detach().numpy()
        m, n = prediction.shape
        num_PI = int(n/2)  #number of PIs

        PI_lower = prediction[:, 0:num_PI]  #lower bound of prediction intervals
        PI_upper = prediction[:, num_PI:]   #upper bound of prediction intervals

        l, u = np.zeros((m, num_PI-1)), np.zeros((m, num_PI-1))
        MUCW, MLCW = 0, 0

        for r in range(num_PI-1):
            l[:, r] = PI_lower[:, r] > PI_lower[:, r+1]
            u[:, r] = PI_upper[:, r] > PI_upper[:, r+1]
            m1 = u[:, r] * (PI_upper[:, r] - PI_upper[:, r+1])
            u1 = u[:, r].sum()
            m2 = l[:, r] * (PI_lower[:, r] - PI_lower[:, r+1])
            l1 = l[:, r].sum()
        
        if u1 == 0:
            MUCW += 0
        else:
            MUCW += m1.sum() / u1

        if l1 == 0:
            MLCW += 0
        else:
            MLCW += m2.sum() / l1
        
        loss = loss_pinball + self.Lambda * l2 + MUCW + MLCW

        return loss