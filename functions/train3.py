
import time
import os
import argparse
import numpy as np
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as LS
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import dataset
import sys
import network
import PIL
from random import *
import pytorch_msssim
from dataset import BSDS500Crop128
import pytorch_ssim

MSE = nn.MSELoss()
#SSIM = pytorch_msssim.SSIM().cuda()
#MSSSIM = pytorch_msssim.MSSSIM().cuda()


class Otimizador():
    adam = 0
    rmsprob = 1
    sgd = 2

class Loss():
    mse = 0
    mae = 1
    crossentropy = 2
    mix_ssim_mse_msssim = 3

def transform_data():       
    train_transform = transforms.Compose([
    transforms.RandomCrop((32, 32)),
    transforms.ToTensor(),])    
    return train_transform


def load_dataset(train_transform):
    train_dataset = BSDS500Crop128(train_path)
    print(len(train_dataset))
    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,num_workers=num_workers,shuffle=True)
    return train_loader

def load_dataset2(train_transform):
    train_dataset = torchvision.datasets.ImageFolder(root=train_path, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,num_workers=num_workers,shuffle=True)
    return train_loader
  
def fuc_scheduler(solver, array_milestones,fator_gamma):
    
    #scheduler = LS.MultiStepLR(solver, milestones=array_milestones, gamma=fator_gamma)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(solver,mode='min',factor=0.65,patience=1,
          verbose=True,)
    return scheduler 

def data_augmentation(patches,r):
    #90º
    if(r<=0.25):
        patches = patches.transpose(2, 3) 
    #180º
    elif(r<=0.5):
        patches = patches.flip(2)
    #270º    
    elif(r<=0.75):
        patches = patches.transpose(2, 3).flip(3)
    return patches

def func_loss(losses, iterations):
    if loss_op == Loss.mse: 
        loss= (sum(losses)/iterations)**2
    elif loss_op == Loss.mae:
        loss= sum(losses)/iterations

    elif loss_op == Loss.mix_ssim_mse_msssim:
        loss= sum(losses)/iterations
        
    return loss    


def define_architecture():
    if cuda:
        encoder = network.EncoderCell().cuda()
        binarizer = network.Binarizer().cuda()
        decoder = network.DecoderCell().cuda()
        gain = network.GainFactor().cuda()

    return encoder,binarizer,decoder,gain

## load networks on GPU
def otimizador_func(encoder,binarizer,decoder):

    if otimizador_op==Otimizador.adam:
        solver = optim.Adam([{'params': encoder.parameters()},
                             {'params': binarizer.parameters()},
                             {'params': decoder.parameters()},],lr=lr)
    return solver

def resume(epoch=None):
    if epoch is None:
        s = 'iter'
        epoch = 0
    else:
        s = 'epoch'
    encoder.load_state_dict(torch.load(path_load+'/encoder_{}_{}.pth'.format(s, epoch)))
    binarizer.load_state_dict(torch.load(path_load+'/binarizer_{}_{}.pth'.format(s, epoch)))
    decoder.load_state_dict(torch.load(path_load+'/decoder_{}_{}.pth'.format(s, epoch)))
    solver.load_state_dict(torch.load(path_load+'/solver_{}_{}.pth'.format(s, epoch)))
    #solver.load_state_dict(torch.load(path_load+'/gain_{}_{}.pth'.format(s, epoch)))


def save(index, epoch=True):
    if not os.path.exists(path_save):
        os.mkdir(path_save)
    if epoch:
        s = 'epoch'
    else:
        s = 'iter'
    torch.save(encoder.state_dict(),path_save+'/encoder_{}_{}.pth'.format(s, index))
    torch.save(binarizer.state_dict(),path_save+'/binarizer_{}_{}.pth'.format(s, index))
    torch.save(decoder.state_dict(), path_save+'/decoder_{}_{}.pth'.format(s, index))
    torch.save(solver.state_dict(), path_save+'/solver_{}_{}.pth'.format(s, index))
    torch.save(gain.state_dict(), path_save+'/gain_{}_{}.pth'.format(s, index))

def compute_psnr(x, y):
    y = y.view(y.shape[0], -1)
    x = x.view(x.shape[0], -1)
    rmse = torch.sqrt(torch.mean((y - x) ** 2, dim=1))
    psnr = torch.mean(20. * torch.log10(1. / rmse))
    return psnr 


batch_size=8  

train_path='/mnt/data/image-databases/database_128x128_HE'
path_save = '/home/marcelo/projetos/checkpoint/db_128x128_HE_mix'
path_load ='/home/marcelo/projetos/checkpoint/cp4_mse'

#train_path='/media/data/Datasets/samsung/database_128x128_HE'
#path_save = '/media/data/Datasets/samsung/modelos/rnn/db_he_ssim'
#path_load ='/media/data/Datasets/samsung/modelos/rnn/cp4_mse_32iter'

data_aug = False
num_workers=5
max_epochs = 20
lr  = 1e-4
cuda =True

iterations = 16
checkpoint = 4

scheduler_op  = True
array_milestones=[3, 10, 20, 50, 100]
fator_gamma=0.5
loss_op = Loss.mix_ssim_mse_msssim 
n_batches_save = 15000 
stop_learning = 3   #numbers os epochs
loss_old = 1
otimizador_op =Otimizador.adam
last_epoch = 0
op_gain =False
k_priming=3

train_transform = transform_data();
train_loader = load_dataset(train_transform)


print('Total de batches:',len(train_loader))
encoder,binarizer,decoder,gain = define_architecture()
solver = otimizador_func(encoder,binarizer,decoder)

if scheduler_op:
   scheduler=fuc_scheduler(solver,array_milestones,fator_gamma)

if checkpoint:
    resume(checkpoint)
    last_epoch = checkpoint
    #if scheduler_op:
     #   scheduler.last_epoch = last_epoch - 1



msssim_epochs = []
loss_epochs = []
ssim_epochs = []
psnr_epochs = []
for epoch in range(last_epoch + 1, max_epochs + 1):
    print('época: ',epoch)
    
    loss_batches = []
    ssim_batches = []
    msssim_batches = []
    psnr_batches = []

    #if scheduler_op:
     #   scheduler.step()
    for batch, data in enumerate(train_loader):

        patches = Variable(data.cuda())
        batch_size, input_channels, height, width = patches.size()
        batch_t0 = time.time()
        ## init lstm state


        encoder_h_1 = (Variable(torch.zeros(batch_size, 256, height // 4, width // 4).cuda()),
                               Variable(torch.zeros(batch_size, 256, height // 4, width // 4)).cuda())
        encoder_h_2 = (Variable(torch.zeros(batch_size, 512, height // 8, width // 8).cuda()),
                               Variable(torch.zeros(batch_size, 512, height // 8, width // 8)).cuda())
        encoder_h_3 = (Variable(torch.zeros(batch_size, 512, height // 16, width // 16).cuda()),
                               Variable(torch.zeros(batch_size, 512, height // 16, width // 16)).cuda())

        decoder_h_1 = (Variable(torch.zeros(batch_size, 512, height // 16, width // 16).cuda()),
                               Variable(torch.zeros(batch_size, 512, height // 16, width // 16)).cuda())
        decoder_h_2 = (Variable(torch.zeros(batch_size, 512, height // 8, width // 8).cuda()),
                               Variable(torch.zeros(batch_size, 512, height // 8, width // 8)).cuda())
        decoder_h_3 = (Variable(torch.zeros(batch_size, 256, height // 4, width // 4).cuda()),
                               Variable(torch.zeros(batch_size, 256, height // 4, width // 4)).cuda())
        decoder_h_4 = (Variable(torch.zeros(batch_size, 128, height // 2, width // 2).cuda()),
                               Variable(torch.zeros(batch_size, 128, height // 2, width // 2)).cuda())

                         
        
        if data_aug:
            r=random()
            patches=data_augmentation(patches,r)


        solver.zero_grad()
        losses = []
        mean_ssim = 0
        mean_msssim = 0
        mean_psnr = 0
        mean_loss = 0
        losses = []

        res = patches - 0.5
        x = res
        bp_t0 = time.time()
        g = 1
        xt = torch.zeros(batch_size, input_channels, height, width,dtype=torch.float).cuda()
        

        masc=0
        for it in range(iterations):
            
            encoded, encoder_h_1, encoder_h_2, encoder_h_3 = encoder(res, encoder_h_1, encoder_h_2, encoder_h_3)
            codes = binarizer(encoded)
            output, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4 = decoder(codes, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4)       
            
            xt = xt+ output
            psnr = compute_psnr(x, xt)
            mse1 = MSE(x,xt)

            ssim = pytorch_ssim.ssim(x,xt) 
            #msssim = MSSSIM(x,xt)
            mse2 = MSE(res, output)

            mean_psnr+=psnr.data.item()/iterations
            mean_ssim+=ssim.data.item()/iterations            

            if loss_op == Loss.mix_ssim_mse_msssim: 
                loss = (1 - ssim) + mse1/5                

            elif loss_op == Loss.mse:
                loss =  mse1
                #(res - output).abs().mean()**2

            elif loss_op == Loss.mae:
                loss = (res - output).abs().mean()
            
            #if psnr>38 or masc==1:
            #    losses=[]
            #    for _ in range(it):
            #        losses.append(torch.zeros(1,dtype=torch.float).cuda())                       
            #    loss = torch.zeros(1,dtype=torch.float).cuda()
            #    masc = 1
            #    print('PSNR:',psnr.data.item())

            mean_loss += loss.data.item()/iterations
            losses.append(loss)
            res = res - output
            
            #if op_gain: 
            #    g = gain(xt)[0].data.item()
            #    ganhos.append(g)

        
        loss = sum(losses)/iterations
        loss.backward()
        solver.zero_grad()
       
        bp_t1 = time.time()
        solver.step()
        batch_t1 = time.time()
        index = (epoch - 1) * len(train_loader) + batch     
        loss_batches.append(mean_loss)
 
        print('[TRAIN] Epoch[{}]({}/{}); Loss média: {:.4f}; Backpropagation: {:.4f} sec; Batch: {:.4f} sec'.format(epoch,batch + 1,len(train_loader), mean_loss, bp_t1 - bp_t0, batch_t1 - batch_t0))
        print(' SSIM: {:.4f} , PSNR: {:.4f}'.format(mean_ssim, mean_psnr))
        print(('{:.4f} ' * iterations +'\n').format(* [l.data.item() for l in losses]))
        
        loss_batches.append(mean_loss)
        ssim_batches.append(mean_ssim)
        #msssim_batches.append(mean_msssim)
        psnr_batches.append(mean_psnr)
        
        #if(index % n_batches_save ==0):   
        #   save(index,False)

      
    save(epoch)
    loss_epochs.append(np.mean(loss_batches))
    ssim_epochs.append(np.mean(ssim_batches))
    #msssim_epochs.append(np.mean(msssim_batches))
    psnr_epochs.append(np.mean(psnr_batches))
    
    np.save('loss_train2',loss_epochs)
    np.save('ssim_train2',ssim_epochs)
    #np.save('msssim',msssim_epochs)
    np.save('psnr_train2',psnr_epochs)
    scheduler.step(loss_epochs[-1])