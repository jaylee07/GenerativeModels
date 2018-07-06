import os, sys, time
import math
import numpy as np
import pickle
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from dataset import get_data
from dataloader import load_dataset
from scipy.misc import imsave

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.utils as utils
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.datasets as dset
import torchvision.utils as v_utils

# Weight initialization
def initialize_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
#         m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Generator(nn.Module):
    def __init__(self, opt, n_channels):
        super(Generator, self).__init__()
        self.opt = opt
        self.n_channels = n_channels
        self.linear_layers = nn.Linear(self.opt.n_z, 8*8*self.opt.channel_bunch)
        layers = nn.Sequential()
        
        layers.add_module("Conv1-1", nn.Conv2d(self.opt.channel_bunch, self.opt.channel_bunch, kernel_size=3, stride=1, padding=1))
        layers.add_module("Activation1-1", nn.ELU(inplace=True))
        layers.add_module("Conv1-2", nn.Conv2d(self.opt.channel_bunch, self.opt.channel_bunch, kernel_size=3, stride=1, padding=1))
        layers.add_module("Activation1-2", nn.ELU(inplace=True))
        layers.add_module("Upsampling1", nn.Upsample(scale_factor=2))
        
        layers.add_module("Conv2-1", nn.Conv2d(self.opt.channel_bunch, self.opt.channel_bunch, kernel_size=3, stride=1, padding=1))
        layers.add_module("Activation2-1", nn.ELU(inplace=True))
        layers.add_module("Conv2-2", nn.Conv2d(self.opt.channel_bunch, self.opt.channel_bunch, kernel_size=3, stride=1, padding=1))
        layers.add_module("Activation2-2", nn.ELU(inplace=True))
        layers.add_module("Upsampling2", nn.Upsample(scale_factor=2))
        
        layers.add_module("Conv3-1", nn.Conv2d(self.opt.channel_bunch, self.opt.channel_bunch, kernel_size=3, stride=1, padding=1))
        layers.add_module("Activation3-1", nn.ELU(inplace=True))
        layers.add_module("Conv3-2", nn.Conv2d(self.opt.channel_bunch, self.opt.channel_bunch, kernel_size=3, stride=1, padding=1))
        layers.add_module("Activation3-2", nn.ELU(inplace=True))
        layers.add_module("Upsampling3", nn.Upsample(scale_factor=2))
        
        layers.add_module("Conv4-1", nn.Conv2d(self.opt.channel_bunch, self.opt.channel_bunch, kernel_size=3, stride=1, padding=1))
        layers.add_module("Activation4-1", nn.ELU(inplace=True))
        layers.add_module("Conv4-2", nn.Conv2d(self.opt.channel_bunch, self.opt.channel_bunch, kernel_size=3, stride=1, padding=1))
        layers.add_module("Activation4-2", nn.ELU(inplace=True))
        
        layers.add_module("Conv5", nn.Conv2d(self.opt.channel_bunch, self.n_channels, kernel_size=3, stride=1, padding=1))
        self.layers = layers
        
    def forward(self, z):
        x = self.linear_layers(z)
        x = x.view(-1, self.opt.channel_bunch, 8, 8)
        x = self.layers(x)
        return x

    
class Decoder(nn.Module):
    def __init__(self, opt, n_channels):
        super(Decoder, self).__init__()
        self.opt = opt
        self.n_channels = n_channels
        self.linear_layers = nn.Linear(self.opt.n_z, 8*8*self.opt.channel_bunch)
        layers = nn.Sequential()
        
        layers.add_module("Conv1-1", nn.Conv2d(self.opt.channel_bunch, self.opt.channel_bunch, kernel_size=3, stride=1, padding=1))
        layers.add_module("Activation1-1", nn.ELU(inplace=True))
        layers.add_module("Conv1-2", nn.Conv2d(self.opt.channel_bunch, self.opt.channel_bunch, kernel_size=3, stride=1, padding=1))
        layers.add_module("Activation1-2", nn.ELU(inplace=True))
        layers.add_module("Upsampling1", nn.Upsample(scale_factor=2))
        
        layers.add_module("Conv2-1", nn.Conv2d(self.opt.channel_bunch, self.opt.channel_bunch, kernel_size=3, stride=1, padding=1))
        layers.add_module("Activation2-1", nn.ELU(inplace=True))
        layers.add_module("Conv2-2", nn.Conv2d(self.opt.channel_bunch, self.opt.channel_bunch, kernel_size=3, stride=1, padding=1))
        layers.add_module("Activation2-2", nn.ELU(inplace=True))
        layers.add_module("Upsampling2", nn.Upsample(scale_factor=2))
        
        layers.add_module("Conv3-1", nn.Conv2d(self.opt.channel_bunch, self.opt.channel_bunch, kernel_size=3, stride=1, padding=1))
        layers.add_module("Activation3-1", nn.ELU(inplace=True))
        layers.add_module("Conv3-2", nn.Conv2d(self.opt.channel_bunch, self.opt.channel_bunch, kernel_size=3, stride=1, padding=1))
        layers.add_module("Activation3-2", nn.ELU(inplace=True))
        layers.add_module("Upsampling3", nn.Upsample(scale_factor=2))
        
        layers.add_module("Conv4-1", nn.Conv2d(self.opt.channel_bunch, self.opt.channel_bunch, kernel_size=3, stride=1, padding=1))
        layers.add_module("Activation4-1", nn.ELU(inplace=True))
        layers.add_module("Conv4-2", nn.Conv2d(self.opt.channel_bunch, self.opt.channel_bunch, kernel_size=3, stride=1, padding=1))
        layers.add_module("Activation4-2", nn.ELU(inplace=True))
        
        layers.add_module("Conv5", nn.Conv2d(self.opt.channel_bunch, self.n_channels, kernel_size=3, stride=1, padding=1))
        self.layers = layers
        
    def forward(self, z):
        x = self.linear_layers(z)
        x = x.view(-1, self.opt.channel_bunch, 8, 8)
        x = self.layers(x)
        return x
    
    
class Encoder(nn.Module):
    def __init__(self, opt, n_channels):
        super(Encoder, self).__init__()
        self.opt = opt
        self.n_channels = n_channels
        layers = nn.Sequential()
        
        layers.add_module("Conv0", nn.Conv2d(self.n_channels, self.opt.channel_bunch, kernel_size=3, stride=1, padding=1))
        
        layers.add_module("Conv1-1", nn.Conv2d(self.opt.channel_bunch, self.opt.channel_bunch, kernel_size=3, stride=1, padding=1))
        layers.add_module("Activation1-1", nn.ELU(inplace=True))
        layers.add_module("Conv1-2", nn.Conv2d(self.opt.channel_bunch, self.opt.channel_bunch, kernel_size=3, stride=1, padding=1))
        layers.add_module("Activation1-2", nn.ELU(inplace=True))
        layers.add_module("Subsampling1", nn.Conv2d(self.opt.channel_bunch, self.opt.channel_bunch, kernel_size=3, stride=2, padding=1))
        layers.add_module("Activation1-3", nn.ELU(inplace=True))
        
        layers.add_module("Conv2-1", nn.Conv2d(self.opt.channel_bunch, self.opt.channel_bunch*2, kernel_size=3, stride=1, padding=1))
        layers.add_module("Activation2-1", nn.ELU(inplace=True))
        layers.add_module("Conv2-2", nn.Conv2d(self.opt.channel_bunch*2, self.opt.channel_bunch*2, kernel_size=3, stride=1, padding=1))
        layers.add_module("Activation2-2", nn.ELU(inplace=True))
        layers.add_module("Subsampling2", nn.Conv2d(self.opt.channel_bunch*2, self.opt.channel_bunch*2, kernel_size=3, stride=2, padding=1))
        layers.add_module("Activation2-3", nn.ELU(inplace=True))
        
        layers.add_module("Conv3-1", nn.Conv2d(self.opt.channel_bunch*2, self.opt.channel_bunch*3, kernel_size=3, stride=1, padding=1))
        layers.add_module("Activation3-1", nn.ELU(inplace=True))
        layers.add_module("Conv3-2", nn.Conv2d(self.opt.channel_bunch*3, self.opt.channel_bunch*3, kernel_size=3, stride=1, padding=1))
        layers.add_module("Activation3-2", nn.ELU(inplace=True))
        layers.add_module("Subsampling3", nn.Conv2d(self.opt.channel_bunch*3, self.opt.channel_bunch*3, kernel_size=3, stride=2, padding=1))
        layers.add_module("Activation3-3", nn.ELU(inplace=True))
        
        layers.add_module("Conv4-1", nn.Conv2d(self.opt.channel_bunch*3, self.opt.channel_bunch*4, kernel_size=3, stride=1, padding=1))
        layers.add_module("Activation4-1", nn.ELU(inplace=True))
        layers.add_module("Conv4-2", nn.Conv2d(self.opt.channel_bunch*4, self.opt.channel_bunch*4, kernel_size=3, stride=1, padding=1))
        layers.add_module("Activation4-2", nn.ELU(inplace=True))

        self.layers = layers
        self.linear_layers = nn.Linear(8*8*self.opt.channel_bunch*4, self.opt.n_z)
    
    def forward(self, x):
        x = self.layers(x)
        x = x.view(-1, 8*8*self.opt.channel_bunch*4)
        z = self.linear_layers(x)
        return z
    
    
class Discriminator(nn.Module):
    def __init__(self, opt, n_channels):
        super(Discriminator, self).__init__()
        self.encoder = Encoder(opt, n_channels)
        self.decoder = Decoder(opt, n_channels)
    def forward(self, x):
        z = self.encoder.forward(x)
        x_hat = self.decoder.forward(z)
        return x_hat
    
    
class BEGAN(object):
    def __init__(self, opt):
        self.opt = opt
        self.trn_loader, self.tst_loader, self.n_channels = load_dataset(self.opt)
        self.is_cuda = torch.cuda.is_available()
        self.lambda_ = 0.001
        self.k = 0.0
        
        self.G = Generator(self.opt, self.n_channels)
        self.D = Discriminator(self.opt, self.n_channels)
        self.G.apply(initialize_weights)
        self.D.apply(initialize_weights)
        self.sample_z = Variable(torch.randn((self.opt.n_sample, self.opt.n_z)), volatile=True)
                
        if self.is_cuda and self.opt.use_cuda:
            self.G, self.D = self.G.cuda(), self.D.cuda()
            self.sample_z = self.sample_z.cuda()
        
        self.optim_G = torch.optim.Adam(self.G.parameters(), lr = self.opt.lrG, betas=(0.5,0.999))
        self.optim_D = torch.optim.Adam(self.D.parameters(), lr = self.opt.lrD, betas=(0.5,0.999))
        
    def train(self):
        self.loss_dict=dict()
        self.loss_dict['G_loss'], self.loss_dict['D_loss'], self.loss_dict['conv_measure'] = list(), list(), list()
        
        # Train Mode
        self.D.train()
        print('------------------Start training------------------')
        for epoch in range(self.opt.maxepoch):
            self.G.train()
            print(">>>>Epoch: {}".format(epoch+1))
            start_time = time.time()
            for iter_num, (image, label) in enumerate(self.trn_loader):
                x = Variable(image)
                z1 = Variable(torch.randn(self.opt.batchsize, self.opt.n_z))
                z2 = Variable(torch.randn(self.opt.batchsize, self.opt.n_z))
                if self.is_cuda:
                    x, z1, z2 = x.cuda(), z1.cuda(), z2.cuda()
                
                # Train D
                self.D.zero_grad()
                D_x = self.D.forward(x)
                L_x = torch.mean(torch.abs(x - D_x))
                
                G_z = self.G.forward(z1)
                D_G_z = self.D.forward(G_z)
                L_Gz = torch.mean(torch.abs(G_z - D_G_z))
                
                D_loss = L_x - self.k * L_Gz
                self.loss_dict['D_loss'].append(D_loss.data[0])
                D_loss.backward()
                self.optim_D.step()
                
                #Train G
                self.G.zero_grad()
                G_z = self.G.forward(z2)
                D_G_z = self.D.forward(G_z)
                L_Gz = torch.mean(torch.abs(G_z - D_G_z))
                G_loss = L_Gz
                
                self.loss_dict['G_loss'].append(G_loss.data[0])
                G_loss.backward()
                self.optim_G.step()
                
                #Convergence Metric(M)
                M = L_x + torch.abs(self.opt.gamma * L_x - L_Gz)
                
                # Update k
                tmp_k = self.k + self.lambda_ * (self.opt.gamma*L_x - L_Gz)
                tmp_k = tmp_k.data[0]
                
                self.k = min(max(tmp_k, 0), 1) # To make 0<=k<=1
                self.M = M.data[0]
                self.loss_dict['conv_measure'].append(M.data[0])

                if (iter_num+1) % 500 == 0:
                    print("Epoch: {}, iter: {}, D_loss: {:.3f}, G_loss: {:.3f}, M: {:.3f}".format(epoch+1, iter_num+1, D_loss.data[0], G_loss.data[0], M.data[0]))
            print(">>>>Time for 1 epoch: {:.2f}".format(time.time()-start_time))
            self.save_results(epoch+1, self.sample_z)
        self.save_model()
            
    def save_results(self, epoch, sample):
        #save result img file
        result_dir = self.opt.result_dir + '/' + self.opt.model + '/' + self.opt.dataset
        exp_config = "channels_{}_dimz_{}_imgsize_{}_batch_{}_gamma_{}".format(self.opt.channel_bunch, self.opt.n_z, self.opt.imagesize, self.opt.batchsize, self.opt.gamma)
        result_dir = os.path.join(result_dir, exp_config)
        
        self.G.eval()
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        fake_file_name = result_dir + '/BEGAN_epoch%03d' %epoch + '.png'
        fake_results = self.G.forward(sample)
        v_utils.save_image(fake_results.data, fake_file_name, nrow = int(math.sqrt(self.opt.n_sample)), normalize=True)

    def save_model(self):
        #save trained models
        save_dir = self.opt.save_dir + '/' + self.opt.model + '/' + self.opt.dataset
        exp_config = "channels_{}_dimz_{}_imgsize_{}_batch_{}_gamma_{}".format(self.opt.channel_bunch, self.opt.n_z, self.opt.imagesize, self.opt.batchsize, self.opt.gamma)
        save_dir = os.path.join(save_dir, exp_config)
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(self.G.state_dict(), os.path.join(save_dir, 'G.pkl'))
        torch.save(self.D.state_dict(), os.path.join(save_dir, 'D.pkl'))
        with open(os.path.join(save_dir, 'loss_dict'), 'wb') as f:
            pickle.dump(self.loss_dict, f)
    
    def load_model(self):
        model_dir = self.opt.save_dir + '/' + self.opt.model + '/' + self.opt.dataset
        exp_config = "channels_{}_dimz_{}_imgsize_{}_batch_{}_gamma_{}".format(self.opt.channel_bunch, self.opt.n_z, self.opt.imagesize, self.opt.batchsize, self.opt.gamma)
        model_dir = os.path.join(model_dir, exp_config)
        
        self.G.load_state_dict(torch.load(os.path.join(model_dir, 'G.pkl')))
        self.D.load_state_dict(torch.load(os.path.join(model_dir, 'D.pkl')))
    