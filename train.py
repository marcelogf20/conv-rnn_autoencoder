
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

class Otimizador():
    adam = 0
    rmsprob = 1
    sgd = 2

class Loss():
    mse = 0
    mae = 1
    crossentropy = 2
    

batch_size=4  
train_path='imagens_teste/'
path_save = 'imagens_codificadas/cod4_mse/'
data_aug =False 
num_workers=5

max_epochs = 6
lr  = 0.0005
cuda =True
iterations = 16
checkpoint = 2
scheduler_op  = True
array_milestones=[3, 10, 20, 50, 100]
fator_gamma=0.5
loss_op = Loss.mse
n_batches_save = 0 
stop_learning = 3   #numbers os epochs
loss_old = 1
otimizador_op =Otimizador.adam
last_epoch = 0


def transform_data():       
    train_transform = transforms.Compose([
    transforms.RandomCrop((32, 32)),
    transforms.ToTensor(),])    
    return train_transform


def load_dataset(train_transform):
    train_dataset = torchvision.datasets.ImageFolder(root=train_path, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,num_workers=num_workers,shuffle=True)
    return train_loader

  
def fuc_scheduler(solver, array_milestones,fator_gamma):
    scheduler = LS.MultiStepLR(solver, milestones=array_milestones, gamma=fator_gamma)
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
    return loss    

def define_architecture():
    if cuda:
        encoder = network.EncoderCell().cuda()
        binarizer = network.Binarizer().cuda()
        decoder = network.DecoderCell().cuda()
    return encoder,binarizer,decoder

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
    encoder.load_state_dict(torch.load(path_save+'/encoder_{}_{}.pth'.format(s, epoch)))
    binarizer.load_state_dict(torch.load(path_save+'/binarizer_{}_{}.pth'.format(s, epoch)))
    decoder.load_state_dict(torch.load(path_save+'/decoder_{}_{}.pth'.format(s, epoch)))
    solver.load_state_dict(torch.load(path_save+'/solver_{}_{}.pth'.format(s, epoch)))


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


train_transform = transform_data();
train_loader=load_dataset(train_transform)

print('Total de batches:',len(train_loader))
encoder,binarizer,decoder = define_architecture()
solver = otimizador_func(encoder,binarizer,decoder)

if scheduler_op:
    scheduler=fuc_scheduler(solver,array_milestones,fator_gamma)

if checkpoint:
    resume(checkpoint)
    last_epoch = checkpoint
    if scheduler_op:
        scheduler.last_epoch = last_epoch - 1


save_loss=[]
loss_atual=[]

for epoch in range(last_epoch + 1, max_epochs + 1):
    if scheduler_op:
        scheduler.step()
    for batch, data in enumerate(train_loader):
        batch_t0 = time.time()
        ## init lstm state
        encoder_h_1 = (Variable(torch.zeros(data[1].size(0), 256, 8, 8).cuda()),
                        Variable(torch.zeros(data[1].size(0), 256, 8, 8)).cuda())
        encoder_h_2 = (Variable(torch.zeros(data[1].size(0), 512, 4, 4).cuda()),
                        Variable(torch.zeros(data[1].size(0), 512, 4, 4).cuda()))
        encoder_h_3 = (Variable(torch.zeros(data[1].size(0), 512, 2, 2).cuda()),
                        Variable(torch.zeros(data[1].size(0), 512, 2, 2).cuda()))

        decoder_h_1 = (Variable(torch.zeros(data[1].size(0), 512, 2, 2).cuda()),
                        Variable(torch.zeros(data[1].size(0), 512, 2, 2).cuda()))
        decoder_h_2 = (Variable(torch.zeros(data[1].size(0), 512, 4, 4).cuda()),
                        Variable(torch.zeros(data[1].size(0), 512, 4, 4).cuda()))
        decoder_h_3 = (Variable(torch.zeros(data[1].size(0), 256, 8, 8).cuda()),
                        Variable(torch.zeros(data[1].size(0), 256, 8, 8).cuda()))
        decoder_h_4 = (Variable(torch.zeros(data[1].size(0), 128, 16, 16).cuda()),
                        Variable(torch.zeros(data[1].size(0), 128, 16, 16).cuda()))

        patches = Variable(data[0].cuda())
        
        if data_aug:
            r=random()
            patches=data_augmentation(patches,r)


        solver.zero_grad()
        losses = []
        res = patches - 0.5
        bp_t0 = time.time()
   
        for _ in range(iterations):
            j+=1
            encoded, encoder_h_1, encoder_h_2, encoder_h_3 = encoder(res, encoder_h_1, encoder_h_2, encoder_h_3)
            codes = binarizer(encoded)
            output, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4 = decoder(codes, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4)
            res = res - output
            losses.append(res.abs().mean())
        bp_t1 = time.time()
        #Função de custo = média do erro absoluto para todas as iterações 
   
        loss = func_loss(losses,iterations)

        #loss = sum(losses)/iterations

        loss.backward()

        solver.step()
        batch_t1 = time.time()
        index = (epoch - 1) * len(train_loader) + batch    
        loss_atual.append(loss.data.item())
        
        ## save checkpoint every 500 training steps
        if n_batches_save:
            if (index+1)%n_batches_save == 0:
                save(index+1,False)    
        

    save_loss.append(np.mean(loss_atual))
    np.save('losses',save_loss)
    save(epoch)


    if(epoch%stop_learning==0):
        print('[TRAIN] Epoch[{}]({}/{}); Loss: {:.6f}; Backpropagation: {:.4f} sec; Batch: {:.4f} sec'.format(epoch,batch + 1,len(train_loader), loss.data.item(), bp_t1 - bp_t0, batch_t1 - batch_t0))
        print(('{:.4f} ' * iterations +'\n').format(* [l.data.item() for l in losses]))
        loss_atual= np.mean(loss_atual)
        #condição de parada: loss não dimnuir a cada 3 épocas que se passam
        if(loss_old<loss_atual):
            np.save('losses',save_loss)
            sys.exit()
        else:
            loss_old=loss_atual
            loss_atual=[]








#train_transform = transforms.Compose([
        #transforms.RandomCrop((32, 32)),
        #transforms.RandomHorizontalFlip(p=0.5),
        #transforms.RandomVerticalFlip(p=0.5),
        #transforms.RandomAffine(degrees=(-10,10), shear=(-10,10), resample=False, fillcolor=40),   
        #transforms.RandomPerspective(distortion_scale=0.2, p=0.35, interpolation=0),
        #transforms.RandomRotation(degrees=(-90,90), resample=PIL.Image.BILINEAR),
        #transforms.ColorJitter(hue=.05, saturation=.05),







