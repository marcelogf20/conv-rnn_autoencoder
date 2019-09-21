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
#import pytorch_msssim
from dataset import BSDS500Crop128, ImageFolderRGB,ImageFolderYCbCr
import pytorch_ssim
from radam import RAdam

MSE = nn.MSELoss()
#SSIM = pytorch_msssim.SSIM().cuda()
#MSSSIM = pytorch_msssim.MSSSIM().cuda()


class Otimizador():
    adam = 0
    rmsprob = 1
    sgd = 2
    radam =3

class Loss():
    mse_rgb = 0
    mae = 1
    ssim_mse = 2
    mse_ycbcr=3
    mae2_ycbcr =4
    mae2_rgb = 5
    mse_l1 = 6
    mse_ycbcr_two_networks =7

def transform_data(size_p):       
    train_transform = transforms.Compose([
    transforms.RandomCrop((size_p, size_p)),
    transforms.ToTensor(),])    
    return train_transform


def load_dataset2(train_transform, type_load):
    
    if type_load =='YCbCr':
        train_dataset = ImageFolderYCbCr(root=train_path, transform=train_transform)
        print('ok')    
    if type_load =='RGB':
        train_dataset = ImageFolderRGB(root=train_path, transform=train_transform)
    
    print(len(train_dataset))
    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,num_workers=num_workers,shuffle=True)
    
    return train_loader
  
def fuc_scheduler(solver, array_milestones,fator_gamma): 
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(solver,mode='min',factor=0.4,patience=10,
          verbose=True)
    return scheduler 
  


def define_architecture():
    if cuda:
        encoder = network.EncoderCell(input_channels=1).cuda()
        binarizer = network.Binarizer(bottleneck=28).cuda()
        decoder = network.DecoderCell(bottleneck=28, output_channels=1).cuda()

        encoder2 = network.EncoderCell(input_channels=2).cuda()
        binarizer2 = network.Binarizer(bottleneck=4).cuda()
        decoder2 = network.DecoderCell(bottleneck=4, output_channels=2).cuda()


    return encoder,binarizer,decoder,encoder2,binarizer2,decoder2

## load networks on GPU
def otimizador_func(encoder,binarizer,decoder,encoder2,binarizer2,decoder2):

    if otimizador_op==Otimizador.adam:
        solver = optim.Adam([{'params': encoder.parameters()},
                             {'params': binarizer.parameters()},
                             {'params': decoder.parameters()},],lr=lr)

        solver2 = optim.Adam([{'params': encoder2.parameters()},
                             {'params': binarizer2.parameters()},
                             {'params': decoder2.parameters()},],lr=lr)


    elif otimizador_op == Otimizador.radam:
        
        solver = RAdam([{'params': encoder.parameters()},
                             {'params': binarizer.parameters()},
                             {'params': decoder.parameters()},],lr=lr)

    return solver,solver2

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
    torch.save(encoder2.state_dict(),path_save+'/encoder2_{}_{}.pth'.format(s, index))
    torch.save(binarizer2.state_dict(),path_save+'/binarizer2_{}_{}.pth'.format(s, index))
    torch.save(decoder2.state_dict(), path_save+'/decoder2_{}_{}.pth'.format(s, index))
    torch.save(solver2.state_dict(), path_save+'/solver2_{}_{}.pth'.format(s, index))

def compute_psnr(x, y):
    y = y.view(y.shape[0], -1)
    x = x.view(x.shape[0], -1)
    rmse = torch.sqrt(torch.mean((y - x) ** 2, dim=1))
    psnr = torch.mean(20. * torch.log10(1. / rmse))
    return psnr 

#path_save = './checkpoint/adam_mse_ycbcr_2networks'
#train_path ='./database/database4'

path_save = './checkpoint/ds4_adam_mse_ycbcr_2networks'
train_path ='./database/database4'



max_epochs, size_patch,batch_size = 1, 32, 32
lr,iterations  = 1e-4,  16
scheduler_op  = False
fator_gamma = 0.5 
n_batches_save = 20000 

otimizador_op = Otimizador.adam
loss_op = Loss.mse_ycbcr_two_networks
type_load ='YCbCr'


num_workers = 8
cuda =True
train_transform = transform_data(size_patch);
train_loader = load_dataset2(train_transform,type_load)

print('Total de batches:',len(train_loader))
encoder,binarizer,decoder,encoder2,binarizer2,decoder2 = define_architecture()
solver,solver2 = otimizador_func(encoder,binarizer,decoder,encoder2,binarizer2,decoder2)


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
        #print('patches.size()',patches.size())

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


        encoder_h_1_2 = (Variable(torch.zeros(batch_size, 256, height // 4, width // 4).cuda()),
                               Variable(torch.zeros(batch_size, 256, height // 4, width // 4)).cuda())
        encoder_h_2_2 = (Variable(torch.zeros(batch_size, 512, height // 8, width // 8).cuda()),
                               Variable(torch.zeros(batch_size, 512, height // 8, width // 8)).cuda())
        encoder_h_3_2 = (Variable(torch.zeros(batch_size, 512, height // 16, width // 16).cuda()),
                               Variable(torch.zeros(batch_size, 512, height // 16, width // 16)).cuda())

        decoder_h_1_2 = (Variable(torch.zeros(batch_size, 512, height // 16, width // 16).cuda()),
                               Variable(torch.zeros(batch_size, 512, height // 16, width // 16)).cuda())
        decoder_h_2_2 = (Variable(torch.zeros(batch_size, 512, height // 8, width // 8).cuda()),
                               Variable(torch.zeros(batch_size, 512, height // 8, width // 8)).cuda())
        decoder_h_3_2 = (Variable(torch.zeros(batch_size, 256, height // 4, width // 4).cuda()),
                               Variable(torch.zeros(batch_size, 256, height // 4, width // 4)).cuda())
        decoder_h_4_2 = (Variable(torch.zeros(batch_size, 128, height // 2, width // 2).cuda()),
                               Variable(torch.zeros(batch_size, 128, height // 2, width // 2)).cuda())



        solver.zero_grad()
        losses = []
        mean_ssim = 0
        mean_msssim = 0
        mean_psnr = 0
        mean_loss = 0
        losses = []
        losses1 = []
        losses2 = []

        patches = patches - 0.5
        res = torch.zeros(batch_size, 1, height, width,dtype=torch.float).cuda()
        
        res[:,0,:,:] = patches[:,0,:,:]
        res2 = patches[:,1:,:,:]
        bp_t0 = time.time()
        g=1
        
        masc=0
        for it in range(iterations):
            
            encoded, encoder_h_1, encoder_h_2, encoder_h_3 = encoder(res, encoder_h_1, encoder_h_2, encoder_h_3) 
            codes = binarizer(encoded)
            output, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4 = decoder(codes, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4)       

            encoded2, encoder_h_1_2, encoder_h_2_2, encoder_h_3_2 = encoder2(res2, encoder_h_1_2, encoder_h_2_2, encoder_h_3_2) 
            codes2 = binarizer2(encoded2)
            output2, decoder_h_1_2, decoder_h_2_2, decoder_h_3_2, decoder_h_4_2 = decoder2(codes2, decoder_h_1_2, decoder_h_2_2, decoder_h_3_2, decoder_h_4_2)       
                      
            #psnr = compute_psnr(x, xt)
            #ssim = pytorch_ssim.ssim(x,xt)
            #msssim =  MSSSIM(x,xt)
            #mse_residual   =   MSE(res, output)
                                
            if loss_op == Loss.ssim_mse:
                mse  = MSE(res, output)
                ssim = pytorch_ssim.ssim(x,xt)

                loss = (1 - ssim) + 0.5*mse
            elif loss_op == Loss.mse_ycbcr_two_networks:
                mse  = MSE(res, output)
                mse2  = MSE(res2, output2)
                #print(res.shape)
                #print(output.shape)
                #print(res2.shape)
                #print(output2.shape)
                loss1 = mse
                loss2 = mse2
                loss = mse+mse2

            elif loss_op == Loss.mse_rgb:  
                mse  = MSE(res, output)
                loss =  mse

            elif loss_op == Loss.mse_l1:
                beta = 0.01   
                mse  = MSE(res, output)
                codes[codes==-1] =0
                l1_loss = torch.norm(codes,p=1)
                print('beta*l1_loss', beta*l1_loss.data.item())
                print('mse', mse.item())
                loss = mse+l1_loss*beta


            elif loss_op == Loss.mae2_rgb:
                loss = (res - output).abs().mean()**2
        
            elif loss_op == Loss.mae:
                loss = (res - output).abs().mean()
            
            
            elif loss_op == Loss.mse_ycbcr:
                mse_y  = MSE(res[:,0,:,:], output[:,0,:,:])
                mse_u  = MSE(res[:,1,:,:], output[:,1,:,:])
                mse_v  = MSE(res[:,2,:,:], output[:,2,:,:])
                loss =  1.3*mse_y + 0.1*mse_u + 0.1*mse_v



            elif loss_op == Loss.mae2_ycbcr:
                mse_y  = (res[:,0,:,:] - output[:,0,:,:]).abs().mean()**2
                mse_u  = (res[:,1,:,:] - output[:,1,:,:]).abs().mean()**2
                mse_v  = (res[:,2,:,:] - output[:,2,:,:]).abs().mean()**2 
                loss =  mse_y + 0.25*mse_u + 0.25*mse_v            
 
            mean_loss += loss.data.item()/iterations
            losses.append(loss)
            losses1.append(loss1)

            losses2.append(loss2)
            res = res - output    
            res2 = res2 - output2
            
        solver.zero_grad()
        loss_batch= sum(losses1)/iterations
        loss_batch.backward(retain_graph=True)
        solver.step()

        solver2.zero_grad()
        loss_batch2= sum(losses2)/iterations
        loss_batch2.backward()
        solver2.step()

        bp_t1 = time.time()
        batch_t1 = time.time()
        index = (epoch - 1) * len(train_loader) + batch     
        loss_batches.append(mean_loss)
 

        print('[TRAIN] Epoch[{}]({}/{}); Loss média: {:.4f}; Backpropagation: {:.4f} sec; Batch: {:.4f} sec'.format(epoch,batch + 1,len(train_loader), mean_loss, bp_t1 - bp_t0, batch_t1 - batch_t0))
        print(('Loss1'+' {:.4f} ' * iterations).format(* [l.data.item() for l in losses1]))
        print(('Loss2'+' {:.4f} ' * iterations).format(* [l.data.item() for l in losses2]))
        print('loss média Y: {:.5f} loss média CbCr {:.5f}'.format( (sum(losses1)/iterations).data.item(), (sum(losses2)/iterations).data.item() ))


        if(index % n_batches_save ==0):   
            save(index,False)

    save(epoch)
    loss_epochs.append(np.mean(loss_batches))
    np.save('loss',loss_epochs)
    #np.save('loss_batches',loss_e)

    #scheduler.step(loss_epochs[-1])
    

