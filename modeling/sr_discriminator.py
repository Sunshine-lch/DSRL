from __future__ import print_function
#%matplotlib inline
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

# Set random seed for reproducibility
manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# custom weights initialization called on netG and netD

class Discriminator(nn.Module):
    def __init__(self, nc, ndf):
        super(Discriminator, self).__init__()
        # self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4

            nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            #
            nn.Conv2d(ndf * 16, ndf * 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 32),
            nn.LeakyReLU(0.2, inplace=True),
            #
            nn.Conv2d(ndf * 32, ndf * 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 64),
            nn.LeakyReLU(0.2, inplace=True),
            #
            nn.Conv2d(ndf * 64, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
        self._init_weight()

    def forward(self, input):
        return self.main(input)
    # def _init_weight(self):
    #     classname = self.__class__.__name__
    #     if classname.find('Conv') != -1:
    #         nn.init.normal_(self.weight.data, 0.0, 0.02)
    #     elif classname.find('BatchNorm') != -1:
    #         nn.init.normal_(self.weight.data, 1.0, 0.02)
    #         nn.init.constant_(m.bias.data, 0)
    def _init_weight(self):
        # classname = self.__class__.__name__
        for m in self.modules():
            if isinstance(m,nn.Conv2d): #isinstance() 函数来判断一个对象是否是一个已知的类型
                nn.init.normal_(m.weight.data, 0.0, 0.02) 
            elif isinstance(m,nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)

def build_sr_discriminator(num_classes,feature_size):
    return Discriminator(num_classes,feature_size)

##--------------------------
# # Number of channels in the training images. For color images this is 3
# nc = 3

# # Size of feature maps in generator
# ngf = 64

# # Size of feature maps in discriminator
# ndf = 64

# # Create the Discriminator
# netD = Discriminator(nc,ndf).to('cuda')

# # Handle multi-gpu if desired

# # if (device.type == 'cuda') and (ngpu > 1):
#     # netD = nn.DataParallel(netD, list(range(ngpu)))

# # Apply the weights_init function to randomly initialize all weights
# #  to mean=0, stdev=0.2.
# # netD.apply(weights_init)

# # Print the model
# print(netD)

# # Training Loop

# # Lists to keep track of progress
# img_list = []
# G_losses = []
# D_losses = []
# iters = 0

# print("Starting Training Loop...")


# # Initialize BCELoss function
# criterion = nn.BCELoss()

# # Create batch of latent vectors that we will use to visualize
# #  the progression of the generator
# # fixed_noise = torch.randn(64, nz, 1, 1, device=device)

# # Establish convention for real and fake labels during training
# real_label = 1
# fake_label = 0

# # Setup Adam optimizers for both G and D
# # optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))

# # For each epoch

# ############################
# # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
# ###########################
# ## Train with all-real batch
# netD.zero_grad()

# # 真实的数据由超分网络给出
# real_cpu = data[0]

# b_size = real_cpu.size(0)
# label = torch.full((b_size,), real_label, device=device)
# # Forward pass real batch through D
# output = netD(real_cpu).view(-1)
# # Calculate loss on all-real batch
# errD_real = criterion(output, label)
# # Calculate gradients for D in backward pass
# errD_real.backward()
# D_x = output.mean().item()

# # noise 由分割网络给出,然后输入进D中
# noise = 0
# fake = netD(noise)
# label.fill_(fake_label)
# # Classify all fake batch with D
# output = netD(fake.detach()).view(-1)
# # Calculate D's loss on the all-fake batch
# errD_fake = criterion(output, label)
# # Calculate the gradients for this batch
# errD_fake.backward()
# D_G_z1 = output.mean().item()
# # Add the gradients from the all-real and all-fake batches

# # 真实的鉴别器，应该是最大化errD,但是这里是将鉴别器替代FAloss，应该取最小
# errD = -1*(errD_real + errD_fake)
# # Update D
# # optimizerD.step()


