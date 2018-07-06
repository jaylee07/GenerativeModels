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
        
class Discriminator(nn.Module):
    def __init__(self, opt, n_channels):
        super(Discriminator, self).__init__()
        self.opt = opt
        self.n_channels = n_channels
        layers = nn.Sequential()
        
        layers.add_module("Conv1", nn.Conv2d(self.n_channels, self.opt.channel_bunch, kernel_size=4, stride=2, padding=1, bias=False))
        layers.add_module("Activation1", nn.LeakyReLU(negative_slope=0.2))
        
        layers.add_module("Conv2", nn.Conv2d(self.opt.channel_bunch, self.opt.channel_bunch*2, kernel_size=4, stride=2, padding=1, bias=False))
        layers.add_module("Batchnorm2", nn.BatchNorm2d(self.opt.channel_bunch*2))
        layers.add_module("Activation2", nn.LeakyReLU(negative_slope=0.2))

        layers.add_module("Conv3", nn.Conv2d(self.opt.channel_bunch*2, self.opt.channel_bunch*4, kernel_size=4, stride=2, padding=1, bias=False)) 
        layers.add_module("Batchnorm3", nn.BatchNorm2d(self.opt.channel_bunch*4))
        layers.add_module("Activation3", nn.LeakyReLU(negative_slope=0.2))

        layers.add_module("Conv4", nn.Conv2d(self.opt.channel_bunch*4, self.opt.channel_bunch*8, kernel_size=4, stride=2, padding=1, bias=False)) 
        layers.add_module("Batchnorm4", nn.BatchNorm2d(self.opt.channel_bunch*8))
        layers.add_module("Activation4", nn.LeakyReLU(negative_slope=0.2))

        layers.add_module("Conv5", nn.Conv2d(self.opt.channel_bunch*8, 1, kernel_size=4, stride=1, padding=0, bias=False))
        layers.add_module("Activation5", nn.Sigmoid())

        self.layers = layers
    
    def forward(self, x):
        if isinstance(x.data, torch.cuda.FloatTensor) and len(self.opt.cuda_index) > 1:
            x = nn.parallel.data_parallel(module=self.layers, inputs=x, device_ids=list(range(len(self.opt.cuda_index.replace(',','')))))
        else:
            x = self.layers(x)
        #x = self.layers(x)
        x = x.view(-1,1).squeeze(1)
        return x
    
    
class Generator(nn.Module):
    def __init__(self, opt, n_channels):
        super(Generator, self).__init__()
        self.opt = opt
        self.n_channels = n_channels
        layers = nn.Sequential()

        layers.add_module("TransConv1", nn.ConvTranspose2d(self.opt.n_z, self.opt.channel_bunch*8, kernel_size=4, stride=1, padding=0, bias=False))
        layers.add_module("Batchnorm1", nn.BatchNorm2d(self.opt.channel_bunch*8))
        layers.add_module("Activation1", nn.ReLU(inplace=True))

        layers.add_module("TransConv2", nn.ConvTranspose2d(self.opt.channel_bunch*8, self.opt.channel_bunch*4, kernel_size=4, stride=2, padding=1, bias=False))
        layers.add_module("Batchnorm2", nn.BatchNorm2d(self.opt.channel_bunch*4))
        layers.add_module("Activation2", nn.ReLU(inplace=True))

        layers.add_module("TransConv3", nn.ConvTranspose2d(self.opt.channel_bunch*4, self.opt.channel_bunch*2, kernel_size=4, stride=2, padding=1, bias=False))
        layers.add_module("Batchnorm3", nn.BatchNorm2d(self.opt.channel_bunch*2))
        layers.add_module("Activation3", nn.ReLU(inplace=True))

        layers.add_module("TransConv4", nn.ConvTranspose2d(self.opt.channel_bunch*2, self.opt.channel_bunch, kernel_size=4, stride=2, padding=1, bias=False))
        layers.add_module("Batchnorm4", nn.BatchNorm2d(self.opt.channel_bunch))
        layers.add_module("Actvation4", nn.ReLU(inplace=True))
    
        layers.add_module("TransConv5", nn.ConvTranspose2d(self.opt.channel_bunch, self.n_channels, kernel_size=4, stride=2, padding=1, bias=False))
        layers.add_module("Actvation5", nn.Tanh())
    
        self.layers = layers
        
    def forward(self, z):
        if isinstance(z.data, torch.cuda.FloatTensor) and len(self.opt.cuda_index) > 1:
            x = nn.parallel.data_parallel(module=self.layers, inputs=z, device_ids=list(range(len(self.opt.cuda_index.replace(',','')))))
        else:
            x = self.layers(z)
        return x
    
    
class DCGAN(object):
    def __init__(self, opt):
        self.opt = opt
        self.trn_loader, self.tst_loader, self.n_channels = load_dataset(opt)
        self.is_cuda = torch.cuda.is_available()

        self.G = Generator(self.opt, self.n_channels)
        self.D = Discriminator(self.opt, self.n_channels)
        self.G.apply(initialize_weights)
        self.D.apply(initialize_weights)
        self.sample_z = Variable(torch.randn((self.opt.n_sample, self.opt.n_z, 1, 1)), volatile=True)
        
        if self.is_cuda and self.opt.use_cuda:
            self.G, self.D = self.G.cuda(), self.D.cuda()
            self.sample_z = self.sample_z.cuda()
        
        self.optim_G = torch.optim.Adam(params=self.G.parameters(), lr=self.opt.lrG, betas=(0.5, 0.999))
        self.optim_D = torch.optim.Adam(params=self.D.parameters(), lr=self.opt.lrD, betas=(0.5, 0.999))
        self.BCEloss = nn.BCELoss()
    
    def train(self):
        self.loss_dict=dict()
        self.loss_dict['G_loss'], self.loss_dict['D_fake_loss'], self.loss_dict['D_real_loss'] = list(), list(), list()

        self.D.train()
        print('------------------Start training------------------')
        for epoch in range(self.opt.maxepoch):
            self.G.train()
            print(">>>>Epoch: {}".format(epoch+1))
            start_time = time.time()
            for iter_num, (image, label) in enumerate(self.trn_loader):
                x = Variable(image)
                z1 = Variable(torch.randn((self.opt.batchsize, self.opt.n_z, 1, 1)))
                z2 = Variable(torch.randn((self.opt.batchsize, self.opt.n_z, 1, 1)))
                true_label = Variable(torch.ones(self.opt.batchsize, 1))
                fake_label = Variable(torch.zeros(self.opt.batchsize, 1))
                if self.is_cuda:
                    x, z1, z2 = x.cuda(), z1.cuda(), z2.cuda()
                    true_label, fake_label = true_label.cuda(), fake_label.cuda()
                
                # Train D
                self.D.zero_grad()
                D_x = self.D.forward(x)
                D_x = D_x.squeeze()
                
                G_z = self.G.forward(z1)
                D_G_z = self.D.forward(G_z)
                D_G_z = D_G_z.squeeze()
                
                D_real_loss = self.BCEloss(D_x, true_label)
                D_fake_loss = self.BCEloss(D_G_z, fake_label)
                D_loss = D_real_loss + D_fake_loss
                D_loss.backward()
                self.optim_D.step()

                #Train G             
                self.G.zero_grad()
                G_z = self.G.forward(z2)
                D_G_z = self.D.forward(G_z)
                D_G_z = D_G_z.squeeze()
                G_loss = self.BCEloss(D_G_z, true_label)
                G_loss.backward()
                self.optim_G.step()

                if (iter_num+1) % 100 == 0:
                    print("Epoch: {}, iter: {}, D_real_loss: {:.3f}, D_fake_loss: {:.3f}, G_loss: {:.3f}".format(epoch+1, iter_num+1, D_real_loss.data[0], D_fake_loss.data[0], G_loss.data[0]))
                    self.loss_dict['G_loss'].append(G_loss.data[0])
                    self.loss_dict['D_fake_loss'].append(D_fake_loss.data[0])
                    self.loss_dict['D_real_loss'].append(D_real_loss.data[0])
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
        fake_file_name = result_dir + '/DCGAN_epoch%03d' %epoch + '.png'
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