
import os
import argparse

import imageio
import numpy as np
from imageio import imread, imsave
import torch
from torch.autograd import Variable
import network

cuda=1
model ='/media/data/Datasets/samsung/modelos/rnn/cp4_masc/'
path_destino='/media/data/Datasets/samsung/resultados/r4_masc'
folder='variaveis'
path_cod = 'imagens_codificadas/cp4_masc/kodim'
    
number_img  = np.load(folder+'/number_img.npy')
iterations   = np.load(folder+'/iterations.npy')
number_model = np.load(folder+'/number_model.npy')

for i in range(number_img,number_img+1):   
    input_npz= path_cod+str(number_img)+'_epoch'+str(number_model)+'_ds4.npz'

    #x=(number_model)*1298+1298
    #if x< 1000:
    #    model=model+'encoder_iter_00000'+str(x)+'.pth'
    #elif x< 10000:
    #    model=model+'encoder_iter_0000'+str(x)+'.pth'
    #elif x<100000:
    #    model=model+'encoder_iter_000'+str(x)+'.pth'
    model=model+'decoder_epoch_'+str(number_model)+'.pth'

    
    decoder = network.DecoderCell()
    decoder.eval()
    decoder.load_state_dict(torch.load(model,map_location=lambda storage, loc: storage))
    
    if iterations==1:
        
        content = np.load(input_npz)
        codes = np.unpackbits(content['codes'])
        codes = np.reshape(codes, content['shape']).astype(np.float32) * 2 - 1
        codes = torch.from_numpy(codes)
        iters, batch_size, channels, height, width = codes.size()
        height = height * 16
        width = width * 16
    
        with torch.no_grad():
            codes = Variable(codes)
            decoder_h_1 = (Variable(torch.zeros(batch_size, 512, height // 16, width // 16)),
                           Variable(torch.zeros(batch_size, 512, height // 16, width // 16)))

            decoder_h_2 = (Variable(torch.zeros(batch_size, 512, height // 8, width // 8)),
                           Variable(torch.zeros(batch_size, 512, height // 8, width // 8)))
            decoder_h_3 = (Variable(torch.zeros(batch_size, 256, height // 4, width // 4)),
                           Variable(torch.zeros(batch_size, 256, height // 4, width // 4)))

            decoder_h_4 = (Variable(torch.zeros(batch_size, 128, height // 2, width // 2)),
                           Variable(torch.zeros(batch_size, 128, height // 2, width // 2)))

            image = torch.zeros(1, 3, height, width) + 0.5
        
    else:
        image = torch.load(folder+'/image.pt')
        codes = torch.load(folder+'/codes.pt')
        output = torch.load(folder+'/output.pt')
        decoder_h_1 = torch.load(folder+'/decoder_h_1.pt')
        decoder_h_2 = torch.load(folder+'/decoder_h_2.pt')
        decoder_h_3=torch.load(folder+'/decoder_h_3.pt')
        decoder_h_4=torch.load(folder+'/decoder_h_4.pt')

        
if cuda:
    decoder = decoder.cuda()
    codes = codes.cuda()
    decoder_h_1 = (decoder_h_1[0].cuda(), decoder_h_1[1].cuda())
    decoder_h_2 = (decoder_h_2[0].cuda(), decoder_h_2[1].cuda())
    decoder_h_3 = (decoder_h_3[0].cuda(), decoder_h_3[1].cuda())
    decoder_h_4 = (decoder_h_4[0].cuda(), decoder_h_4[1].cuda())    

    passo=1
    #if iterations==16:
    #    passo=1

    for iters in range(iterations,iterations+passo):
        output, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4 = decoder(
                codes[iters-1], decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4)
        image = image + output.data.cpu()
        #name_img_out = 'kodim'+str(number_img)+'_epoch'+str(number_model)+'_iter'
        #imageio.imwrite(os.path.join(path_destino,name_img_out+'{}.bmp'.format(iters)),np.squeeze(image.numpy().clip(0, 1) * 255.0).astype(np.uint8).transpose(1, 2, 0))


    torch.save(image,folder+ '/image.pt')
    torch.save(codes,folder+ '/codes.pt')
    torch.save(output,folder+ '/output.pt')
    torch.save(decoder_h_1,folder+ '/decoder_h_1.pt')
    torch.save(decoder_h_2, folder+'/decoder_h_2.pt')
    torch.save(decoder_h_3, folder+'/decoder_h_3.pt')
    torch.save(decoder_h_4, folder+'/decoder_h_4.pt')
    np.save(folder+'/iterations',iterations+passo)

    if iters==16:
        name_img_out = 'kodim'+str(number_img)+'_epoch'+str(number_model)+'_iter'
        print('Modelo:',number_model,'; Imagem:',number_img,'; Iteração:',iters)
        imageio.imwrite(os.path.join(path_destino,name_img_out+'{:02d}.bmp'.format(iters)),np.squeeze(image.numpy().clip(0, 1) * 255.0).astype(np.uint8).transpose(1, 2, 0))
        iterations=1
        np.save(folder+'/iterations',iterations)
        if(number_model==17):
            number_model=17
            number_img+=1
            np.save(folder+'/number_model',number_model)
            np.save(folder+'/number_img',number_img)
        else:
            number_model+=1
            np.save(folder+'/number_model',number_model)