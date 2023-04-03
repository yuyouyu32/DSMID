import torch
from torch.nn import init

class Parameters(object):
    def __init__(self, batchSize=512, imageSize=(24, 21), nz=512, ngf=64, ndf=64, 
                       nepochs=1500, lr=0.0001, beta1=0.8, gpu=1, adv_weight=0.1,
                       lrd=0.0001, alpha=0.3, outf='results', f_emb=8) -> None:
        self.batchSize=batchSize
        self.imageSize=imageSize
        self.nz=nz      # size of the latent z vector
        self.ngf=ngf    # Number of filters to use in the generator network
        self.ndf=ndf    # Number of filters to use in the discriminator network
        self.nepochs=nepochs
        self.lr=lr
        self.beta1=beta1    # beta1 for adam. default=0.5
        self.gpu=gpu        # GPU to use, -1 for CPU training
        self.adv_weight=adv_weight  # weight for adv loss
        self.lrd=lrd    # learning rate decay, default=0.0001
        self.alpha=alpha    # multiplicative factor for target adv. loss
        self.outf=outf

def weights_init(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		m.weight.data.normal_(0.0, 0.02)
	elif classname.find('BatchNorm') != -1:
		m.weight.data.normal_(1.0, 0.02)
		m.bias.data.fill_(0)
	elif classname.find('Linear') != -1:
		size = m.weight.size()
		m.weight.data.normal_(0.0, 0.1)
		m.bias.data.fill_(0)

def weights_init_xavier(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		init.xavier_normal(m.weight.data, gain=0.02)
	elif classname.find('Linear') != -1:
		init.xavier_normal(m.weight.data, gain=0.02)
	elif classname.find('BatchNorm2d') != -1:
		init.normal(m.weight.data, 1.0, 0.02)
		init.constant(m.bias.data, 0.0)

def lr_scheduler(optimizer, lr):
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr
	return optimizer

def exp_lr_scheduler(optimizer, epoch, init_lr, lrd, nevals):
    """Implements torch learning reate decay with SGD"""
    lr = init_lr / (1 + nevals*lrd)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer
