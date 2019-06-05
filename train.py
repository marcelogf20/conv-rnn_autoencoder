
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
import torch.utils.data as data2
from torchvision import transforms

batch_size=32    
training=['DatasetsEduardo/database4']

path_saved= ['checkpoint/checkpoint4']
max_epochs =100
lr=0.0001
cuda=True
iterations=16
checkpoint=0
import torchvision
import torchvision.transforms as transforms
import dataset
import sys
for train,path_save in zip(training,path_saved):

    train_transform = transforms.Compose([
    transforms.RandomCrop((32, 32)),
    transforms.ToTensor(),])

    def load_dataset():
        train_dataset = torchvision.datasets.ImageFolder(
        root=train,
        transform=train_transform)
        print(len(train_dataset))
        train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=32,num_workers=12,shuffle=True)
        return train_loader

    train_loader=load_dataset()
 
            
    print('total batches:',len(train_loader))

    ## load networks on GPU
    import network

    encoder = network.EncoderCell().cuda()
    binarizer = network.Binarizer().cuda()
    decoder = network.DecoderCell().cuda()
    solver = optim.Adam(
        [
            {
                'params': encoder.parameters()
            },
            {
                'params': binarizer.parameters()
            },
            {
                'params': decoder.parameters()
            },
        ],
        lr=lr)
    def resume(epoch=None):
        if epoch is None:
            s = 'iter'
            epoch = 0
        else:
            s = 'epoch'

        encoder.load_state_dict(
            torch.load(path_save+'/encoder_{}_{:08d}.pth'.format(s, epoch)))
        binarizer.load_state_dict(
            torch.load(path_save+'/binarizer_{}_{:08d}.pth'.format(s, epoch)))
        decoder.load_state_dict(
            torch.load(path_save+'/decoder_{}_{:08d}.pth'.format(s, epoch)))

    def resume2():
        encoder.load_state_dict(
            torch.load('checkpoint/checkpoint4/decoder_iter_00077880.pth'))
        binarizer.load_state_dict(
            torch.load('checkpoint/checkpoint4/binarizer_iter_00077880.pth'))
        decoder.load_state_dict(
             torch.load('checkpoint/checkpoint4/encoder_iter_00077880.pth'))


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


    # resume()

    #scheduler = LS.MultiStepLR(solver, milestones=[3, 10, 20, 50, 100], gamma=0.5)
    last_epoch = 0
    if checkpoint:
        resume(checkpoint)
        last_epoch = checkpoint
#        scheduler.last_epoch = last_epoch - 1
    loss_old = 1
    save_loss=[]
    loss_atual=[]
    for epoch in range(last_epoch + 1, max_epochs + 1):
     #   scheduler.step()
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

            solver.zero_grad()

            losses = []

            res = patches - 0.5

            bp_t0 = time.time()

            for _ in range(iterations):
                encoded, encoder_h_1, encoder_h_2, encoder_h_3 = encoder(res, encoder_h_1, encoder_h_2, encoder_h_3)
                codes = binarizer(encoded)

                output, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4 = decoder(
                    codes, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4)

                res = res - output
                losses.append(res.abs().mean())

            bp_t1 = time.time()

            loss = sum(losses)/iterations
            loss.backward()
            solver.step()
            batch_t1 = time.time()
            index = (epoch - 1) * len(train_loader) + batch
            
            loss_atual.append(loss.data.item())
            ## save checkpoint every 500 training steps
            if (batch+1)%9000 == 0:
                save(batch+1,False)    
        

        save_loss.append(np.mean(loss_atual))
        save(epoch)
        np.save('losses',save_loss)
        if( (epoch)%3):
            print('[TRAIN] Epoch[{}]({}/{}); Loss: {:.6f}; Backpropagation: {:.4f} sec; Batch: {:.4f} sec'.format(epoch,batch + 1,len(train_loader), loss.data.item(), bp_t1 - bp_t0, batch_t1 - batch_t0))
            print(('{:.4f} ' * iterations +'\n').format(* [l.data.item() for l in losses]))
            loss_atual= np.mean(loss_atual)
            if(loss_old<loss_atual):
                np.save('losses',save_loss)
                sys.exit()
            else:
                loss_old=loss_atual
                loss_atual=[]
















