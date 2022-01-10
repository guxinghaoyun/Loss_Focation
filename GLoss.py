import torch
from torch.autograd import Variable
from torch.autograd import Function
from torch.autograd.function import once_differentiable

class GL(Function):
    @staticmethod
    def forward(self,inputx,input_lab,cens,variances, numclasses = 10):
        if cens.size(0) != numclasses or variances.size(0)!=numclasses:
            raise ValueError("The number of centers or variances does not equal to classes.")

        if inputx.is_cuda:
            loss = torch.zeros(1).cuda()
            diff = torch.zeros_like(inputx).cuda()
        else:
            loss = torch.zeros(1)
            diff = torch.zeros_like(inputx)
        #input_lab = input_lab.long()
        num_input_class = torch.bincount(input_lab)
        num_input_class = num_input_class.float()
        centers_pred = cens.index_select(0,input_lab)
        var_pred = variances.index_select(0,input_lab)
        count_class = num_input_class.index_select(0,input_lab)
        diff = inputx - centers_pred
        loss = 1/2.0*((diff.pow(2) / var_pred).sum(1) / count_class).sum()
        
        diff = diff / var_pred / count_class.reshape(diff.size(0),1)

        self.save_for_backward(diff)

        return loss
    
    @staticmethod
    @once_differentiable
    def backward(self,output_grad):
        diff, = self.save_tensors
        input_grad = diff * output_grad
        return input_grad, None, None, None, None