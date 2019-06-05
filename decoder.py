
import os
import argparse

import imageio
import numpy as np
from imageio import imread, imsave
import torch
from torch.autograd import Variable
import network
cuda=1

m='./checkpoint/checkpoint_2_4/decoder_iter_000'
    
folder='./resultados/resultados_2_4'

    
number_img  = np.load('variaveis/number_img.npy')
iterations   = np.load('variaveis/iterations.npy')
number_model = np.load('variaveis/number_model.npy')


for i in range(number_img,number_img+1):
    input_npz='./imagens_codificadas/cod_2_4/kodim'+str(number_img)+'_model'+str(number_model)+'_ds_2_4.npz'
 
    x=(number_model)*1299 +1298
    if x< 1000:
        model=m+'00'+str(x)+'.pth'
    elif x< 10000:
        model=m+'0'+str(x)+'.pth'
    else:
        model=m+str(x)+'.pth'
    

    
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
        image = torch.load('variaveis/image.pt')
        codes = torch.load('variaveis/codes.pt')
        output = torch.load('variaveis/output.pt')
        decoder_h_1 = torch.load('variaveis/decoder_h_1.pt')
        decoder_h_2 = torch.load('variaveis/decoder_h_2.pt')
        decoder_h_3=torch.load('variaveis/decoder_h_3.pt')
        decoder_h_4=torch.load('variaveis/decoder_h_4.pt')

        
if cuda:
    decoder = decoder.cuda()
    codes = codes.cuda()
    decoder_h_1 = (decoder_h_1[0].cuda(), decoder_h_1[1].cuda())
    decoder_h_2 = (decoder_h_2[0].cuda(), decoder_h_2[1].cuda())
    decoder_h_3 = (decoder_h_3[0].cuda(), decoder_h_3[1].cuda())
    decoder_h_4 = (decoder_h_4[0].cuda(), decoder_h_4[1].cuda())    

    passo=5 
    if iterations==11:
        passo=6

    for iters in range(iterations,iterations+passo):
        output, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4 = decoder(
                codes[iters-1], decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4)
        image = image + output.data.cpu()
    torch.save(image, 'variaveis/image.pt')
    torch.save(codes, 'variaveis/codes.pt')
    torch.save(output, 'variaveis/output.pt')
    torch.save(decoder_h_1, 'variaveis/decoder_h_1.pt')
    torch.save(decoder_h_2, 'variaveis/decoder_h_2.pt')
    torch.save(decoder_h_3, 'variaveis/decoder_h_3.pt')
    torch.save(decoder_h_4, 'variaveis/decoder_h_4.pt')
    np.save('variaveis/iterations',iterations+passo)

    if iters==16:
	name_img_out = 'kodim'+str(number_img)+'_model'+str(number_model)+'_iter'
        print('Modelo:',number_model,'; Imagem:',number_img,'; Iteração:',iters)
        imageio.imwrite(os.path.join(folder,name_img_out+'{:02d}.bmp'.format(iters)),np.squeeze(image.numpy().clip(0, 1) * 255.0).astype(np.uint8).transpose(1, 2, 0))
        iterations=1
        np.save('variaveis/iterations',iterations)
        if(number_model==59):
            number_model=0
            number_img+=1
            np.save('variaveis/number_model',number_model)
            np.save('variaveis/number_img',number_img)
        else:
            number_model+=1
            np.save('variaveis/number_model',number_model)

