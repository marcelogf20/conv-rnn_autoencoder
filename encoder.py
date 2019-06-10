import argparse

import numpy as np
from scipy.misc import imread, imresize, imsave

import torch
from torch.autograd import Variable



model ='checkpoint/checkpoint4_2epochs/'
inputs ='imagens_teste/'
path_destino= 'imagens_codificadas/cod4_2epochs/'
#iterations=16
cuda=0

number_img   = np.load('variaveis/number_img.npy')
iterations   = np.load('variaveis/iterations.npy')
number_model = np.load('variaveis/number_model.npy')

for i in range(number_img,number_img+1):   

    if i<10:
        input_img =inputs+'kodim0'+str(i)+'.bmp'
    else:
        input_img =inputs+'kodim'+str(i)+'.bmp'

    x=(number_model)*1298+1298
    if x< 1000:
        model=model+'encoder_iter_00000'+str(x)+'.pth'
    elif x< 10000:
        model=model+'encoder_iter_0000'+str(x)+'.pth'
    elif x<100000:
        model=model+'encoder_iter_000'+str(x)+'.pth'



    
    image = imread(input_img, pilmode='RGB')
    image = torch.from_numpy(
        np.expand_dims(
            np.transpose(image.astype(np.float32) / 255.0, (2, 0, 1)), 0))
    batch_size, input_channels, height, width = image.size()
    assert height % 32 == 0 and width % 32 == 0
    with torch.no_grad():
        image = Variable(image)

    encoder = network.EncoderCell()
    binarizer = network.Binarizer()
    decoder = network.DecoderCell()
  

    encoder.load_state_dict(torch.load(model,map_location=lambda storage, loc: storage ) )
    binarizer.load_state_dict(torch.load(model.replace('encoder', 'binarizer'),map_location=lambda storage, loc: storage ))
    decoder.load_state_dict(torch.load(model.replace('encoder', 'decoder'),map_location=lambda storage, loc: storage))


    if iterations > 1:
        with torch.no_grad():
                 
            encoder_h_1 = torch.load('variaveis/encoder_h_1.pt')
            encoder_h_2 = torch.load('variaveis/encoder_h_2.pt')
            encoder_h_3 = torch.load('variaveis/encoder_h_3.pt')
            decoder_h_1 = torch.load('variaveis/decoder_h_1.pt')
            decoder_h_2 = torch.load('variaveis/decoder_h_2.pt')
            decoder_h_3=torch.load('variaveis/decoder_h_3.pt')
            decoder_h_4=torch.load('variaveis/decoder_h_4.pt')
       
        with open("variaveis/codes.txt", "rb") as fp:   # Unpickling
            codes = pickle.load(fp)
        
        encoded=torch.load('variaveis/encoded.pt')
        res = torch.load('variaveis/res.pt')
        output = torch.load('variaveis/output.pt')

    else:
        codes = []
        res = image - 0.5
        with torch.no_grad():  
            encoder_h_1 = (Variable(
                    torch.zeros(batch_size, 256, height // 4, width // 4)),
                               Variable(
                                   torch.zeros(batch_size, 256, height // 4, width // 4)))
            encoder_h_2 = (Variable(
                    torch.zeros(batch_size, 512, height // 8, width // 8)),
                               Variable(
                                   torch.zeros(batch_size, 512, height // 8, width // 8)))
            encoder_h_3 = (Variable(
                    torch.zeros(batch_size, 512, height // 16, width // 16)),
                               Variable(
                                   torch.zeros(batch_size, 512, height // 16, width // 16)))

            decoder_h_1 = (Variable(
                    torch.zeros(batch_size, 512, height // 16, width // 16)),
                               Variable(
                                   torch.zeros(batch_size, 512, height // 16, width // 16)))
            decoder_h_2 = (Variable(
                    torch.zeros(batch_size, 512, height // 8, width // 8)),
                               Variable(
                                   torch.zeros(batch_size, 512, height // 8, width // 8)))
            decoder_h_3 = (Variable(
                    torch.zeros(batch_size, 256, height // 4, width // 4)),
                               Variable(
                                   torch.zeros(batch_size, 256, height // 4, width // 4)))
            decoder_h_4 = (Variable(
                    torch.zeros(batch_size, 128, height // 2, width // 2)),
                               Variable(
                                   torch.zeros(batch_size, 128, height // 2, width // 2)))

                         
      
    if cuda:
        
        if iterations > 1:
            image = image.cuda()
            res = image - 0.5
        
        encoder = encoder.cuda()
        binarizer = binarizer.cuda()
        decoder = decoder.cuda()
        encoder_h_1 = (encoder_h_1[0].cuda(), encoder_h_1[1].cuda())
        encoder_h_2 = (encoder_h_2[0].cuda(), encoder_h_2[1].cuda())
        encoder_h_3 = (encoder_h_3[0].cuda(), encoder_h_3[1].cuda())

        decoder_h_1 = (decoder_h_1[0].cuda(), decoder_h_1[1].cuda())
        decoder_h_2 = (decoder_h_2[0].cuda(), decoder_h_2[1].cuda())
        decoder_h_3 = (decoder_h_3[0].cuda(), decoder_h_3[1].cuda())
        decoder_h_4 = (decoder_h_4[0].cuda(), decoder_h_4[1].cuda())    
    
        
    passo=5
    if iterations==11:
        passo=6
    for iters in range(iterations,iterations+passo):
        encoded, encoder_h_1, encoder_h_2, encoder_h_3 = encoder(
            res, encoder_h_1, encoder_h_2, encoder_h_3)
        code = binarizer(encoded)

        output, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4 = decoder(
            code, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4)

        res = res - output
        codes.append(code.data.cpu().numpy())


    torch.save(encoded, 'variaveis/encoded.pt')
    torch.save(encoder_h_1, 'variaveis/encoder_h_1.pt')
    torch.save(encoder_h_2, 'variaveis/encoder_h_2.pt')
    torch.save(encoder_h_3, 'variaveis/encoder_h_3.pt')
    torch.save(res, 'variaveis/res.pt')
    torch.save(output, 'variaveis/output.pt')
    torch.save(decoder_h_1, 'variaveis/decoder_h_1.pt')
    torch.save(decoder_h_2, 'variaveis/decoder_h_2.pt')
    torch.save(decoder_h_3, 'variaveis/decoder_h_3.pt')
    torch.save(decoder_h_4, 'variaveis/decoder_h_4.pt')
    with open("variaveis/codes.txt", "wb") as fp:   #Pickling
        pickle.dump(codes, fp)
    np.save('variaveis/iterations',iterations+passo)
    
    if iters==16:
        print('Img: {}; Model {}; Iter: {:02d}; Loss: {:.06f}'.format(number_img, number_model,iters, res.data.abs().mean()))
        codes = (np.stack(codes).astype(np.int8) + 1) // 2
        export = np.packbits(codes.reshape(-1))
        name_cod =path_destino+'kodim'+str(number_img)+'_model'+str(number_model)+'_ds_2_4.npz'

        np.savez_compressed(str(name_cod), shape=codes.shape, codes=export)
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
    
 


