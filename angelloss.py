import torch
import math

def Radius(ndim):
    if ndim<1:
        raise ValueError("The dimension should be more than or equal 1.")
    ndim = int(ndim)

    if ndim ==1:
        return 0
    elif ndim == 2:
        return 0.5
    else:        
        return 0.5/math.sqrt(1-(Radius(ndim-1))**2)

def costheta(ndim):
    if ndim<2:
        raise ValueError("The dimension should be more than or equal to 2.")
    return 2*Radius(ndim-1)**2-1

class AngelLoss(torch.nn.Module):    
    def __init__(self, num_classes, feat_dim, cenloss_weight=1.0,angloss_weight=1.0):        
        super(AngelLoss, self).__init__()        
        self.num_classes = num_classes        
        self.feat_dim = feat_dim       
        self.cenloss_weight = cenloss_weight
        self.angloss_weight =  angloss_weight
        self.centers = torch.nn.Parameter(torch.randn(num_classes, feat_dim).cuda())
        self.costheta =  costheta(ndim=self.num_classes)
        
    def forward(self, y, feat):
        # cenloss_weight = 0,   only angle loss
        # angloss_weight = 0,   only center loss        
        batch_size = feat.size()[0]
        if feat.size()[1] != self.feat_dim:  
            raise ValueError("Center's dim: {0} should be equal to input feature's dim: {1}"
            .format(self.feat_dim,feat.size()[1])) 
        if self.angloss_weight == 0:            
            feat = feat.view(batch_size, 1, 1, -1).squeeze()     
            centers_pred = self.centers.index_select(0, y.long())  
            diff = feat-centers_pred 
            loss = 0.5 *self.cenloss_weight * diff.pow(2).sum()/batch_size  #center loss
            
        elif self.cenloss_weight == 0:            
            loss = 0
            dist = self.centers.data.norm(p=2,dim=1)
            for i in range(self.num_classes):
                for j in range(self.num_classes):
                    if i!=j:
                        loss += (self.centers[i].matmul(self.centers[j].data)/(dist[i]*dist[j]) - self.costheta)**2
            loss /= self.angloss_weight * 0.5*self.num_classes*(self.num_classes-1)     # Angel loss
            
        else:
            feat = feat.view(batch_size, 1, 1, -1).squeeze()       
            centers_pred = self.centers.index_select(0, y.long())        
            diff = feat-centers_pred 
            loss_cen = 0.5 * diff.pow(2).sum()/batch_size  #center loss


            loss_ang = 0
            dist = self.centers.data.norm(p=2,dim=1)
            for i in range(self.num_classes):
                for j in range(self.num_classes):
                    if i!=j:
                        loss_ang += (self.centers[i].matmul(self.centers[j].data)/(dist[i]*dist[j]) - self.costheta)**2
            loss_ang /= 0.5*self.num_classes*(self.num_classes-1)          #Angle loss

            loss = loss_cen*self.cenloss_weight + loss_ang*self.angloss_weight
        
        return loss