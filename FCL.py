import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as TF
import torchvision.utils as vutils
import torch.nn.functional as F
from torch.autograd import Function


class FixedCenterLoss(nn.Module):
    """
    paper: http://ydwen.github.io/papers/WenECCV16.pdf
    code:  https://github.com/pangyupo/mxnet_center_loss
    pytorch code: https://blog.csdn.net/sinat_37787331/article/details/80296964
    """

    def __init__(self, num_class, fixed_weights, loss_weight=0.005):
        """
        初始化        
        :param num_class: 类别数量
        :param fixed_weights 固定权值向量
        :param loss_weight:   centerloss的权重系数 [0,1]
        """
        assert 0 <= loss_weight <= 1
        super(FixedCenterLoss, self).__init__()
        self.loss_weight = loss_weight
        self.num_class = num_class
        self.weights = fixed_weights      
        # store the center of each class , should be ( num_class, features_dim)
        self.centers_gamma = nn.Parameter(torch.ones(num_class,1))
        self.lossfunc = CenterLossFunc.apply
        #init_weight(self, 'normal')

    def forward(self, output_features, y_truth):
        """
        损失计算
        :param output_features: linear features [b, -1]
        :param y_truth:  标签值  [b,]
        :return:
        """
        #batch_size = y_truth.size(0)
        #output_features = output_features.view(batch_size, -1)        
        #assert output_features.size(-1) == self.feat_dim
        
        loss = self.lossfunc(output_features, y_truth, self.weights, self.centers_gamma)
        loss *= self.loss_weight

        # centers_pred = self.feature_centers.index_select(0, y_truth.long())  # [b,features_dim]
        # diff = output_features - centers_pred
        # loss = self.alpha * 1 / 2.0 * (diff.pow(2).sum()) / self.batch_size
        return loss


class CenterLossFunc(Function):
    # https://blog.csdn.net/xiewenbo/article/details/89286462
    @staticmethod
    def forward(ctx, feat, labels, fixed_weights, weights_gamma):
        ctx.save_for_backward(feat, labels, fixed_weights, weights_gamma)
        centers = fixed_weights*weights_gamma
        centers_batch = centers.index_select(0, labels.long())
        return (feat - centers_batch).pow(2).sum() * 0.5 / labels.size(0)

    @staticmethod
    def backward(ctx, grad_output):        
        feature, label, center_weights, center_gamma = ctx.saved_tensors  
        centers = center_gamma * center_weights      
        centers_batch = centers.index_select(0, label.long())
        #print(centers_batch)
        diff = (feature - centers_batch) / label.size(0)
        # init every iteration
        counts = centers.new(center_gamma.size(0)).fill_(1e-6)
        ones = centers.new(label.size(0)).fill_(1)

        tmp = centers.new(centers.size()).fill_(0)
        #grad_gamma = centers.new(center_gamma.size()).fill_(0)              

        counts = counts.scatter_add_(0, label.long(), ones)

        tmp.scatter_add_(0, label.unsqueeze(
            1).expand(feature.size()).long(), feature)        
        #print(tmp.size())
        #print(center_weights.size())

        grad_gamma = (tmp * center_weights).sum(dim=1) / counts      
        
        grad_centers = center_gamma - grad_gamma.view(-1,1)
        #print(grad_gamma)
        #print(center_gamma)
        #print(grad_centers)
        #grad_centers = grad_centers / counts.view(-1, 1)
        #print(grad_centers)
        return  grad_output * diff, None, None, grad_centers/label.size(0)

