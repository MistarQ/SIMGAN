import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class ConvBlock(nn.Sequential):
    def __init__(self, in_channel, out_channel, ker_size, padd, stride):
        super(ConvBlock,self).__init__()
        self.add_module('conv',nn.Conv2d(in_channel ,out_channel,kernel_size=ker_size,stride=stride,padding=padd)),
        self.add_module('norm',nn.BatchNorm2d(out_channel)),
        self.add_module('LeakyRelu',nn.LeakyReLU(0.2, inplace=True))

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('Norm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
   


class ConditionalWDiscriminator(nn.Module):
    def __init__(self, opt):
        super(ConditionalWDiscriminator, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        N = int(opt.nfc)
        self.head = ConvBlock(opt.nc_im,N,opt.ker_size,opt.padd_size,1)
        self.body = nn.Sequential()
        for i in range(opt.num_layer-2):
            N = int(opt.nfc/pow(2,(i+1)))
            block = ConvBlock(max(2*N,opt.min_nfc),max(N,opt.min_nfc),opt.ker_size,opt.padd_size,1)
            self.body.add_module('block%d'%(i+1),block)
        self.tail = nn.Conv2d(max(N,opt.min_nfc),1,kernel_size=opt.ker_size,stride=1,padding=opt.padd_size)
        self.generatelabel = nn.Conv2d(max(N,opt.min_nfc),2,kernel_size=opt.ker_size,bias=False)
        self.generatelabel = nn.Sequential(
            nn.Conv2d(max(N,opt.min_nfc),2,kernel_size=opt.ker_size,bias=False),
            nn.Softmax(dim=1)
        )
        # Future works:
        # 1. make sure that the ouput of the self.body is fixed, and obtain its dimdension to compute the flattened dimension 
        # 2. Improve the parameterization of tmp_net below
        # Note: if the output of self.generatelabel is to be processed with F.cross_entrorpy, then do not append activation layer.
        #tmp_net = [ nn.Flatten(start_dim=1), nn.Linear( max(N,opt.min_nfc), 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 2)]
        #self.generatelabel = nn.Sequential( *tmp_net)
        
    def forward(self,x):
        x = self.head(x)
        x = self.body(x)
        c = self.generatelabel(x)
        c = torch.mean(c, dim=3)
        #print(c)
        c = torch.mean(c, dim=2)
        #print(c)
        x = self.tail(x)
        return x,c

class WDiscriminator(nn.Module):
    def __init__(self, opt):
        super(WDiscriminator, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        N = int(opt.nfc)
        self.head = ConvBlock(opt.nc_im,N,opt.ker_size,opt.padd_size,1)
        self.body = nn.Sequential()
        for i in range(opt.num_layer-2):
            N = int(opt.nfc/pow(2,(i+1)))
            block = ConvBlock(max(2*N,opt.min_nfc),max(N,opt.min_nfc),opt.ker_size,opt.padd_size,1)
            self.body.add_module('block%d'%(i+1),block)
        self.tail = nn.Conv2d(max(N,opt.min_nfc),1,kernel_size=opt.ker_size,stride=1,padding=opt.padd_size)

    def forward(self,x):
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        return x    

# 条件G, y1 = G(x,c), y= a1 * y2 + (1-a1) * y2
class ConditionalGeneratorConcatSkip2CleanAddAlpha(nn.Module):
    def __init__(self, opt):
        super(ConditionalGeneratorConcatSkip2CleanAddAlpha, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        N = opt.nfc
        self.head = ConvBlock(opt.nc_im+2,N,opt.ker_size,opt.padd_size,1)
        self.body = nn.Sequential()
        for i in range(opt.num_layer-2):
            N = int(opt.nfc/pow(2,(i+1)))
            block = ConvBlock(max(2*N,opt.min_nfc),max(N,opt.min_nfc),opt.ker_size,opt.padd_size,1)
            self.body.add_module('block%d'%(i+1),block)
        self.tail = nn.Sequential(
            nn.Conv2d(max(N,opt.min_nfc),opt.nc_im,kernel_size=opt.ker_size,stride =1,padding=opt.padd_size),
            nn.Tanh()
        )
        N = opt.nfc
        self.head2 = ConvBlock(opt.nc_im*3+2,N,opt.ker_size,opt.padd_size,1) 
        self.body2 = nn.Sequential()
        for i in range(2):
            N = int(opt.nfc/pow(2,(i+1)))
            block = ConvBlock(max(2*N,opt.min_nfc),max(N,opt.min_nfc),opt.ker_size,opt.padd_size,1)
            self.body.add_module('block%d'%(i+1),block)
        self.tail2 = nn.Sequential(
            nn.Conv2d(max(N,opt.min_nfc),opt.nc_im,kernel_size=opt.ker_size,stride =1,padding=opt.padd_size),
            nn.Sigmoid()
        )
    def forward(self,x,y2,c):
        
        c=torch.Tensor(c)
        c=c.cuda()
        c=c.view(1,c.size(0),1,1)
        c=c.repeat(1,1,x.size(2),x.size(3))
        x=torch.cat([x,c],dim=1)
        #print(c.shape,x.shape)     
         
       
        y1 = self.head(x)
        y1 = self.body(y1)
        y1 = self.tail(y1)
        ind = int((y2.shape[2]-y1.shape[2])/2)
        y2 = y2[:,:,ind:(y2.shape[2]-ind),ind:(y2.shape[3]-ind)]
        x_c = torch.cat((x,y1,y2),1)
        #print(y1.shape)
        #print(ind)
        #print(y2.shape)
        #print(x_c.shape) 
       
        a1 = self.head2(x_c)
        a1 = self.body2(a1)
        a1 = self.tail2(a1)
        #print(a1.shape)       
   
        return a1*y2 + (1-a1)*y1
    
    
# y= a1 * y2 + (1-a1) * y2   
class GeneratorConcatSkip2CleanAddAlpha(nn.Module):
    def __init__(self, opt):
        super(GeneratorConcatSkip2CleanAddAlpha, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        N = opt.nfc
        self.head = ConvBlock(opt.nc_im,N,opt.ker_size,opt.padd_size,1)
        self.body = nn.Sequential()
        for i in range(opt.num_layer-2):
            N = int(opt.nfc/pow(2,(i+1)))
            block = ConvBlock(max(2*N,opt.min_nfc),max(N,opt.min_nfc),opt.ker_size,opt.padd_size,1)
            self.body.add_module('block%d'%(i+1),block)
        self.tail = nn.Sequential(
            nn.Conv2d(max(N,opt.min_nfc),opt.nc_im,kernel_size=opt.ker_size,stride =1,padding=opt.padd_size),
            nn.Tanh()
        )
        N = opt.nfc
        self.head2 = ConvBlock(opt.nc_im*3,N,opt.ker_size,opt.padd_size,1) 
        self.body2 = nn.Sequential()
        for i in range(2):
            N = int(opt.nfc/pow(2,(i+1)))
            block = ConvBlock(max(2*N,opt.min_nfc),max(N,opt.min_nfc),opt.ker_size,opt.padd_size,1)
            self.body.add_module('block%d'%(i+1),block)
        self.tail2 = nn.Sequential(
            nn.Conv2d(max(N,opt.min_nfc),opt.nc_im,kernel_size=opt.ker_size,stride =1,padding=opt.padd_size),
            nn.Sigmoid()
        )
    def forward(self,x,y2):
        y1 = self.head(x)
        y1 = self.body(y1)
        y1 = self.tail(y1)
        ind = int((y2.shape[2]-y1.shape[2])/2)
        y2 = y2[:,:,ind:(y2.shape[2]-ind),ind:(y2.shape[3]-ind)]
        x_c = torch.cat((x,y1,y2),1)
        
        a1 = self.head2(x_c)
        a1 = self.body2(a1)
        a1 = self.tail2(a1)
        
        return a1*y2 + (1-a1)*y1    
