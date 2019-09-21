from math import log, e
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
<<<<<<< HEAD
from dataset import BSDS500Crop128, ImageFolderRGB,ImageFolderYCbCr
=======
from dataset import BSDS500Crop128
>>>>>>> faf252d1a292b5c48a1de45f68064d177c46a3e4
import pytorch_ssim
from radam import RAdam

MSE = nn.MSELoss()
#SSIM = pytorch_msssim.SSIM().cuda()
#MSSSIM = pytorch_msssim.MSSSIM().cuda()

<<<<<<< HEAD
class HLoss(nn.Module):
    def __init__(self):
        super(HLoss, self).__init__()

    def forward(self, x):
        b = F.softmax(x,dim=1) * F.log_softmax(x,dim=1)
        b = -1.0 * b.sum()
        return b

criterion = HLoss()

=======
>>>>>>> faf252d1a292b5c48a1de45f68064d177c46a3e4

class Otimizador():
    adam = 0
    rmsprob = 1
    sgd = 2
    radam =3

class Loss():
    mse_rgb = 0
    mae = 1
<<<<<<< HEAD
    ssim_mse = 2
    mse_ycbcr=3
    mae2_ycbcr =4
    mae2_rgb =5
    mse_l1 = 6
    mseRGB_patches =8


=======
    ssim_mse = 3
    mse_ycbcr=4

>>>>>>> faf252d1a292b5c48a1de45f68064d177c46a3e4
def transform_data(size_p):       
    train_transform = transforms.Compose([
    transforms.RandomCrop((size_p, size_p)),
    transforms.ToTensor(),])    
    return train_transform


<<<<<<< HEAD

def load_dataset2(train_transform, type_load='RGB'):
    
    #if type_load =='YCbCr':
    #    train_dataset = ImageFolderYCbCr(root=train_path, transform=train_transform)
    
    if type_load =='RGB':
        train_dataset = ImageFolderRGB(root=train_path, transform=train_transform)
    
=======
def load_dataset(train_transform):
    
    train_dataset = BSDS500Crop128(train_path, train_transform)
>>>>>>> faf252d1a292b5c48a1de45f68064d177c46a3e4
    print(len(train_dataset))
    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,num_workers=num_workers,shuffle=True)
    
    return train_loader
<<<<<<< HEAD
  
def fuc_scheduler(solver, array_milestones,fator_gamma): 
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(solver,mode='min',factor=0.4,patience=10,
          verbose=True)
=======

def load_dataset2(train_transform):
    train_dataset = torchvision.datasets.ImageFolder(root=train_path, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,num_workers=num_workers,shuffle=True)
    return train_loader
  
def fuc_scheduler(solver, array_milestones,fator_gamma): 
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(solver,mode='min',factor=0.4,patience=10,
          verbose=True,)
>>>>>>> faf252d1a292b5c48a1de45f68064d177c46a3e4
    return scheduler 
  


def define_architecture():
    if cuda:
<<<<<<< HEAD
        encoder = network.EncoderCell(3).cuda()
        binarizer = network.Binarizer(32).cuda()
        decoder = network.DecoderCell(32,3).cuda()
=======
        encoder = network.EncoderCell().cuda()
        binarizer = network.Binarizer().cuda()
        decoder = network.DecoderCell().cuda()
>>>>>>> faf252d1a292b5c48a1de45f68064d177c46a3e4

    return encoder,binarizer,decoder

## load networks on GPU
def otimizador_func(encoder,binarizer,decoder):

    if otimizador_op==Otimizador.adam:
        solver = optim.Adam([{'params': encoder.parameters()},
                             {'params': binarizer.parameters()},
                             {'params': decoder.parameters()},],lr=lr)

    elif otimizador_op == Otimizador.radam:
        
        solver = RAdam([{'params': encoder.parameters()},
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

def compute_psnr(x, y):
    y = y.view(y.shape[0], -1)
    x = x.view(x.shape[0], -1)
    rmse = torch.sqrt(torch.mean((y - x) ** 2, dim=1))
    psnr = torch.mean(20. * torch.log10(1. / rmse))
    return psnr 


<<<<<<< HEAD
 

train_path ='/media/data/Datasets/samsung/database4'
path_save  = '/media/data/Datasets/samsung/modelos/rnn/adam_msergb_patches'

#path_load ='/media/data/Datasets/samsung/modelos/rnn/cp4_mse_32iter'

#path_save = './checkpoint/ds4_adam_mse_rgb'
#train_path ='./database/database4'


max_epochs = 1
size_patch = 32
batch_size = 32

lr  = 7e-4
iterations = 16
scheduler_op  = False
fator_gamma = 0.5 
n_batches_save =111100 

otimizador_op = Otimizador.adam
loss_op = Loss.mseRGB_patches
type_load ='RGB'



num_workers = 8
cuda =True
train_transform = transform_data(size_patch);
train_loader = load_dataset2(train_transform,type_load)
=======
batch_size = 32  
train_path ='/media/data/Datasets/samsung/database4/dir0'
path_save  = '/media/data/Datasets/samsung/modelos/rnn/mse_ycbcr'
#path_load ='/media/data/Datasets/samsung/modelos/rnn/cp4_mse_32iter'

data_aug = False
num_workers = 5

max_epochs = 2

lr  = 1e-4
cuda =True
iterations = 16
checkpoint = 0
last_epoch = 0

scheduler_op  = False
fator_gamma = 0.5 
n_batches_save = 15000 
loss_old = 1

otimizador_op = Otimizador.radam
loss_op = Loss.mse_ycbcr 

size_patch = 32

train_transform = transform_data(size_patch);
train_loader = load_dataset(train_transform)
>>>>>>> faf252d1a292b5c48a1de45f68064d177c46a3e4

print('Total de batches:',len(train_loader))
encoder,binarizer,decoder = define_architecture()
solver = otimizador_func(encoder,binarizer,decoder)

op_target_quality = False
data_aug = False
loss_old = 1
target_quality = 42
last_epoch = 0
checkpoint = 0

if scheduler_op:
   scheduler=fuc_scheduler(solver,array_milestones,fator_gamma)

if checkpoint:
    resume(checkpoint)
    last_epoch = checkpoint
    #if scheduler_op:
     #   scheduler.last_epoch = last_epoch - 1
<<<<<<< HEAD
=======


>>>>>>> faf252d1a292b5c48a1de45f68064d177c46a3e4

msssim_epochs=[]
loss_epochs=[]
ssim_epochs=[]
psnr_epochs=[]

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
<<<<<<< HEAD
=======
        #print('patches.size()',patches.size())
>>>>>>> faf252d1a292b5c48a1de45f68064d177c46a3e4

        batch_size, input_channels, height, width = patches.size()
        batch_t0 = time.time()
        ## init lstm state
<<<<<<< HEAD
=======


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
>>>>>>> faf252d1a292b5c48a1de45f68064d177c46a3e4

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

        solver.zero_grad()
        losses = []
        mean_ssim = 0
        mean_msssim = 0
        mean_psnr = 0
        mean_loss = 0
        losses = []

        res = patches - 0.5
        x=res
        bp_t0 = time.time()
        g=1
        xt = torch.zeros(batch_size, input_channels, height, width,dtype=torch.float).cuda()
        
        masc=0
        for it in range(iterations):
            
            encoded, encoder_h_1, encoder_h_2, encoder_h_3 = encoder(res, encoder_h_1, encoder_h_2, encoder_h_3)
            codes = binarizer(encoded)
            output, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4 = decoder(codes, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4)       
            xt = xt + output
<<<<<<< HEAD

            #psnr = compute_psnr(x, xt)
            #ssim = pytorch_ssim.ssim(x,xt)
            #msssim =  MSSSIM(x,xt)
            #mse_residual   =   MSE(res, output)

            if loss_op == Loss.ssim_mse:
                mse_recons = MSE(x,xt,)
                loss = (1 - ssim) + mse_recons/5                 
            
            elif loss_op == Loss.mse_rgb:  
                mse  = MSE(res, output)
                loss =  mse
            elif loss_op == Loss.mseRGB_patches:
                mse  = MSE(xt,x)
                loss =  mse

            elif loss_op == Loss.mse_l1:
                ent, base, beta = 0, 2, 0.01
                
                mse  = MSE(res, output)
                index = (epoch - 1) * len(train_loader) + batch     
                c = codes.clone()
                #ent = criterion(c)
                     
                #values = c.unique(sorted=True).type(torch.cuda.FloatTensor)
                values = torch.tensor([-1.0,1.0],dtype=torch.float).cuda() 
                counts = torch.stack([(c==c_u).sum() for c_u in values]).type(torch.cuda.FloatTensor)
                n_labels = counts[0]+counts[1]
                probs = counts/n_labels
                
                base = 2 
                for p in probs:
                    ent -= p * log(p, base) 
               
                l1_loss = ent

                if it ==15:
                    print('counts',counts[0].data.item(), counts[1].data.item())
                    print('l1_loss *beta',beta * l1_loss.data.item())        
                    print('mse', mse.item())
               
                loss = 0*mse + l1_loss*beta


            elif loss_op == Loss.mae2_rgb:
                loss = (res - output).abs().mean()**2
            
            elif loss_op == Loss.mse_ycbcr:
                mse_y  = MSE(res[:,0,:,:], output[:,0,:,:])
                mse_u  = MSE(res[:,1,:,:], output[:,1,:,:])
                mse_v  = MSE(res[:,2,:,:], output[:,2,:,:])
                loss =  mse_y + 0.25*mse_u + 0.25*mse_v

            elif loss_op == Loss.mae2_ycbcr:
                mse_y  = (res[:,0,:,:] - output[:,0,:,:]).abs().mean()**2
                mse_u  = (res[:,1,:,:] - output[:,1,:,:]).abs().mean()**2
                mse_v  = (res[:,2,:,:] - output[:,2,:,:]).abs().mean()**2 
                loss =  mse_y + 0.25*mse_u + 0.25*mse_v            
=======

            #psnr = compute_psnr(x, xt)
            #ssim = pytorch_ssim.ssim(x,xt)
            #msssim =  MSSSIM(x,xt)
            #mse_residual   =   MSE(res, output)

            if loss_op == Loss.ssim_mse:
                mse_recons = MSE(x,xt,)
                loss = (1 - ssim) + mse_recons/5                 
            
            elif loss_op == Loss.mse_rgb:
                mse_recons = MSE(x,xt)
                loss =  mse_recons
                #(res - output).abs().mean()**2

            elif loss_op == Loss.mae:
                loss = (res - output).abs().mean()
            
            elif loss_op == Loss.mse_ycbcr:
                mse_y  = MSE(res[:,0,:,:], output[:,0,:,:])
                mse_u  = MSE(res[:,1,:,:], output[:,1,:,:])
                mse_v  = MSE(res[:,2,:,:], output[:,2,:,:])
                loss =  mse_y + 0.25*mse_u + 0.25*mse_v

            
>>>>>>> faf252d1a292b5c48a1de45f68064d177c46a3e4
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
            
<<<<<<< HEAD
        
        solver.zero_grad()
        loss_batch= sum(losses)/iterations
        
        #if(op_target_quality):
         #   psnr = compute_psnr(x, xt)
         #   if psnr<target_quality:
         #       loss_batch.backward()
         #       solver.step()
         #   else:
         #       print('Batch com PSNR (dB) de: %.4f'%(psnr.data.item()))
        #else:
=======
    
        solver.zero_grad()
        loss_batch= sum(losses)/iterations
>>>>>>> faf252d1a292b5c48a1de45f68064d177c46a3e4
        loss_batch.backward()
        solver.step()

        bp_t1 = time.time()
        batch_t1 = time.time()
        index = (epoch - 1) * len(train_loader) + batch     
        loss_batches.append(mean_loss)
 
        print('[TRAIN] Epoch[{}]({}/{}); Loss média: {:.4f}; Backpropagation: {:.4f} sec; Batch: {:.4f} sec'.format(epoch,batch + 1,len(train_loader), mean_loss, bp_t1 - bp_t0, batch_t1 - batch_t0))
        #print(' SSIM: {:.4f} , MSSSIM: {:.4f}, PSNR: {:.4f}'.format(mean_ssim, mean_msssim, mean_psnr))
        print(('{:.4f} ' * iterations +'\n').format(* [l.data.item() for l in losses]))
<<<<<<< HEAD

        if(index % n_batches_save ==0):   
            save(index,False)

    save(epoch)
    loss_epochs.append(np.mean(loss_batches))
    np.save('loss',loss_epochs)
    #np.save('loss_batches',loss_e)

    #scheduler.step(loss_epochs[-1])
    
=======

        #if(index % n_batches_save ==0):   
        #    save(index,False)

    save(epoch)
    loss_epochs.append(np.mean(loss_batches))
    np.save('loss',loss_epochs)
    #scheduler.step(loss_epochs[-1])
    
>>>>>>> faf252d1a292b5c48a1de45f68064d177c46a3e4
