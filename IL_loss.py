import torch

class ILloss(torch.nn.Module):    
    def __init__(self, num_classes, feat_dim, lamb=0.01, lamb1=10):        
        super(ILloss, self).__init__()        
        self.num_classes = num_classes        
        self.feat_dim = feat_dim  
        self.lamb = lamb     
        self.lamb1 = lamb1
        self.centers = torch.nn.Parameter(torch.randn(num_classes, feat_dim).cuda())
        
    def forward(self, y, feat):     
        batch_size = feat.size()[0]
        if feat.size()[1] != self.feat_dim:  
            raise ValueError("Center's dim: {0} should be equal to input feature's dim: {1}"
            .format(self.feat_dim,feat.size()[1])) 

        feat = feat.view(batch_size, 1, 1, -1).squeeze()       
        centers_pred = self.centers.index_select(0, y.long())        
        diff = feat-centers_pred 
        loss_cen = 0.5 * diff.pow(2).sum()/batch_size  #center loss


        loss_ang = 0
        dist = self.centers.data.norm(p=2,dim=1)
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                if i!=j:
                    loss_ang += self.centers[i].matmul(self.centers[j].data)/(dist[i]*dist[j]) + 1
                #Angle loss

        loss = self.lamb*(loss_cen + loss_ang*self.lamb1)
        
        return loss