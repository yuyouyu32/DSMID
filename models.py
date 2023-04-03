import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.autograd import Function

"""
Reverse Layer
"""
class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None


"""
CORAL
"""
def CORAL(source, target, DEVICE):
    d = source.size(1)
    ns, nt = source.size(0), target.size(0)

    # source covariance
    tmp_s = torch.ones((1, ns)).double().to(DEVICE) @ source
    cs = (source.t() @ source - (tmp_s.t() @ tmp_s) / ns) / (ns - 1)

    # target covariance
    tmp_t = torch.ones((1, nt)).double().to(DEVICE) @ target
    ct = (target.t() @ target - (tmp_t.t() @ tmp_t) / nt) / (nt - 1)

    # frobenius norm
    loss = (cs - ct).pow(2).sum().sqrt()
    loss = loss / (4 * d * d)

    return loss


"""
Generator network
"""
class _netG(nn.Module):
    def __init__(self, opt, nclasses):
        super(_netG, self).__init__()
        
        self.ndim = opt.ndf
        self.ngf = opt.ngf
        self.nz = opt.nz
        self.gpu = opt.gpu
        self.nclasses = nclasses
        
        self.main = nn.Sequential(
        nn.ConvTranspose2d(self.nz+self.ndim+self.nclasses+1, self.ngf, 2, 1, 0, bias=True), # (2, 2)
        nn.BatchNorm2d(self.ngf, affine=True),
        # nn.LeakyReLU(0.2, inplace=True),

        nn.ConvTranspose2d(self.ngf, (self.ngf // 2), 4, 2, 1, bias=True),    # (4, 4)
        nn.BatchNorm2d((self.ngf // 2), affine=True),
        # nn.LeakyReLU(0.2, inplace=True),

        nn.ConvTranspose2d((self.ngf // 2), (self.ngf // 4), (4,3), 2, 1, bias=True),    # （8，7）
        nn.BatchNorm2d((self.ngf // 4), affine=True),
        # nn.LeakyReLU(0.2, inplace=True),

        nn.ConvTranspose2d((self.ngf // 4), 1, 4, 3, 1, output_padding=1, bias=True),
        nn.Tanh()
        )

    def forward(self, input):   
        batchSize = input.size()[0]
        input = input.view(-1, self.ndim+self.nclasses+1, 1, 1)
        noise = torch.FloatTensor(batchSize, self.nz, 1, 1).normal_(0, 1)    
        if self.gpu>=0:
            noise = noise.cuda()
        noisev = Variable(noise)
        output = self.main(torch.cat((input, noisev),1))
        return output

"""
Discriminator network
"""
class _netD(nn.Module):
    def __init__(self, opt, nclasses):
        super(_netD, self).__init__()
        self.nclasses = nclasses
        self.ndf = opt.ndf
        self.feature = nn.Sequential(
            nn.Conv2d(1, self.ndf, kernel_size=3, stride=3),    # (8, 7)
            # nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2, ceil_mode=True),                 # (4, 4)

            nn.Conv2d(self.ndf, self.ndf*2, kernel_size=3, stride=1),    # (2, 2)
            # nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 1, ceil_mode=True),                          # (1, 1)           
        )
        self.f_emb = nn.Linear(self.ndf * 2, self.ndf)
        self.classifier_c = nn.Sequential(
                            nn.Linear(self.ndf, (self.ndf // 2)),
                            nn.LeakyReLU(0.2, inplace=True),
                            nn.Linear((self.ndf // 2), self.nclasses))              
        self.classifier_s = nn.Sequential(
        						nn.Linear(self.ndf, (self.ndf // 2)),
                                nn.LeakyReLU(0.2, inplace=True),
                                nn.Linear((self.ndf // 2), 1),
        						nn.Sigmoid())              

    def forward(self, input, f_emb, f_emb_rvs=None):
               
        output = self.feature(input)
        output = self.f_emb(output.view(-1, self.ndf*2))
        output_s = self.classifier_s(output).view(-1)
        output_c = self.classifier_c(output)
        output_c_f = self.classifier_c(f_emb)
        if f_emb_rvs is None:
            output_s_f = self.classifier_s(f_emb).view(-1)
            return output_s, output_c, output_s_f, output_c_f 
        else:
            output_s_f_rev = self.classifier_s(f_emb).view(-1)
            return output_s, output_c, output_s_f_rev, output_c_f
            
"""
Feature extraction network
"""
class _netF(nn.Module):
    def __init__(self, opt):
        super(_netF, self).__init__()

        self.ndf = opt.ndf
        self.feature = nn.Sequential(
            nn.Conv2d(1, self.ndf, kernel_size=3, stride=3),    # (8, 7)
            # nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2, ceil_mode=True),                 # (4, 4)

            nn.Conv2d(self.ndf, self.ndf*2, kernel_size=3, stride=1),    # (2, 2)
            # nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 1, ceil_mode=True),                          # (1, 1)
        )
        self.emb = nn.Linear(2*self.ndf, self.ndf)


    def forward(self, input, reverse_grad=False, alpha=0):   
        output = self.feature(input)
        output =self.emb(output.view(-1, 2*self.ndf))
        if reverse_grad:
            output = ReverseLayerF.apply(output, alpha)
        return output

"""
Classifier network
"""
class _netC(nn.Module):
    def __init__(self, opt, nclasses):
        super(_netC, self).__init__()
        self.ndf = opt.ndf
        self.main = nn.Sequential(          
            nn.Linear(self.ndf, ((self.ndf) // 2)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(((self.ndf) // 2), nclasses),                         
        )

    def forward(self, input):       
        output = self.main(input)
        return output

