import torch
from torch.nn.parameter import Parameter

#have potential bug that occurs when batch_size>1
Tensor = torch.cuda.FloatTensor
class DLoss(torch.nn.Module):
    def __init__(self,shape):
        super(DLoss,self).__init__()
        # self.subscale=int(1/subscale)
        self.real_label = Parameter(Tensor(shape,).fill_(1.0), requires_grad=False)
        self.fake_label = Parameter(Tensor(shape,).fill_(0.0), requires_grad=False)
        self.criterion = torch.nn.BCELoss()
    def forward(self,real_output,generator_output):
        b_size = real_output.size(0)
        # label = torch.full((b_size,),self.real_label,dtype=torch.float,device='cuda:0')
        real_output = real_output.view(-1)
        errD_real = self.criterion(real_output,self.real_label)

        # label.fill_(self.fake_label)
        generator_output = generator_output.view(-1)
        errD_fake = self.criterion(generator_output,self.fake_label)
        D_loss = (errD_fake + errD_real)/2
        # return errD_real,errD_fake
        return D_loss
