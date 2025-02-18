

num_img = '*'

path_save = 'kodim'+str(num_img)+'_reconstruida.png'

path_load = 'imagens_teste/kodim'+str(num_img)+'.bmp'
#path_model = '/media/data/Datasets/samsung/modelos/rnn/mse_ycbcr/encoder_epoch_1.pth'

path_model = './checkpoint/ds4_adam_mse_ycbcr_2networks/encoder_epoch_1.pth'
path_model2 = './checkpoint/ds4_adam_mse_ycbcr_2networks/encoder2_epoch_1.pth'

path_destino = 'resultados'
input_channels = 3
size_patch = 32
batch_size = 4

target_psnr = 32
min_iters = 1
qiters=16

colorspace_input = 'YCbCr'

height = size_patch
width  = size_patch
cuda   = 1
op_bit_allocation =  False
type_psnr ='Y'


import time,bitstring
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import dataset
import network
import PIL
import sys
from PIL import Image
from pathlib import PosixPath
import pickle
from skimage.util import view_as_blocks
from img_common.msssim import compare_msssim
from skimage.measure import compare_psnr, compare_ssim
import glob,os,gzip


def calc_metric(true_ref,test_ref,metric):
  
  true_ref = np.array(true_ref)
  test_ref = np.array(test_ref)
  
  if metric=='psnr':
    result = compare_psnr(true_ref, test_ref) 
  elif metric =='ssim':
    result = compare_ssim(true_ref, test_ref, multichannel=True)
  elif metric =='msssim':
    true_ref = np.expand_dims(np.expand_dims(true_ref, 0), 0).swapaxes(0, -1)
    test_ref = np.expand_dims(np.expand_dims(test_ref, 0), 0).swapaxes(0, -1)
    aux_result = list(map(compare_msssim, true_ref, test_ref))
    result = np.mean(aux_result)
      
  # Put a huge number in this case
  result = np.iinfo(np.uint8).max if result == float('inf') else result
  return result


class Manipulation_Images():

    def pad_img(self,img, patch_size):
        """ Method that receives an image and a size for the patches. The method
            pad the image so that they can be cropped later
        """
        orig_shape = np.array(img.shape[:2])
        new_shape = patch_size * np.ceil(orig_shape / patch_size).astype(int)
        points_to_pad = new_shape - orig_shape
        pad_img = np.pad(img, [(0, points_to_pad[0]), (0, points_to_pad[1]),
                             (0, 0)], 'edge')
        return pad_img

    def extract_img_patch(self,orig_img, patch_size):
        
        """ Method that receives an image and the patch size and extract
            the patches of the image.
        """

        if np.all(np.equal(orig_img.shape, patch_size)):
            return orig_img

        img = self.pad_img(orig_img, patch_size)
        color = 1
        if len(img.shape) > 2:
            color = img.shape[2]
        patches = view_as_blocks(img, (patch_size, patch_size, color))
        patches = patches.reshape(-1, patch_size, patch_size, color)
        return patches
                  
          
    def conv_data_format(self, img, data_format):
        if not isinstance(data_format, ImgData):
            raise ValueError("Format argument must be an " + ImgData.__name__)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')

            if data_format == ImgData.UBYTE:
                out_img = img_as_ubyte(img)
                return out_img

            if data_format == ImgData.FLOAT:
                out_img = img_as_float(img)
                out_img = out_img.astype(np.float32)
                return out_img

        raise ValueError('Range parameter is not recognized!')

            
    def load_image(self,path, cm):

        valid, img = self.is_pillow_valid_img(path)
        if not valid:
            return []
        if cm:
            img = img.convert(cm)

        return np.array(img)

    def is_pillow_valid_img(self,pathname, return_ref=True):

        try:
            img = Image.open(pathname)
            is_valid = True
        except Exception:
            img = []
            is_valid = False

        ret = is_valid
        if return_ref:
            ret = list([ret])
            ret.append(img)
        else:
            img.close()

        return ret
          
    def save_img(self,img_ref, path, mode='RGB'):
        try:
            if isinstance(img_ref, (str, PosixPath)):
                img_ref = Image.open(img_ref)
            elif isinstance(img_ref, np.ndarray):
                img_ref = Image.fromarray(img_ref)
                print('here')

            img_ref.save(path, mode=mode)
            img_ref.close()
        except Exception as e:
            print(str(e))      
        return img_ref
          
          
    def calc_metric(self,true_ref,test_ref,metric):
        true_ref = np.array(true_ref)
        test_ref = np.array(test_ref)
        if metric=='psnr':
            result = compare_psnr(true_ref, test_ref)  
        elif metric =='ssim':
            result = compare_ssim(true_ref, test_ref, multichannel=True)
        elif metric =='msssim':
            true_ref = np.expand_dims(np.expand_dims(true_ref, 0), 0).swapaxes(0, -1)
            test_ref = np.expand_dims(np.expand_dims(test_ref, 0), 0).swapaxes(0, -1)
            aux_result = list(map(compare_msssim, true_ref, test_ref))
            result = np.mean(aux_result)
      
        # Put a huge number in this case
        result = np.iinfo(np.uint8).max if result == float('inf') else result
        return result


    def reconstruct_image(self,patches, width, height):
        """ Method that receives a 1D array of patches and the dimensions of
            the image and reconstruct the image based on the patches array
        """
        patch_size = patches[0].shape[0]
        colors = 1
        if len(patches[0].shape) > 2:
            colors = patches.shape[3]
        p_rows = np.ceil(height / patch_size).astype(int)
        p_columns = np.ceil(width / patch_size).astype(int)
        round_height = p_rows * patch_size
        round_width = p_columns * patch_size

        # First index changing fastest (reshape a grid of patches)
        img = patches.reshape(p_rows, p_columns, patch_size, patch_size,
                                  colors)
        # Rows and columns near one another (for numpy to order them correctly)
        img = img.swapaxes(1, 2).reshape(round_height, round_width, colors)
        img = img[:height, :width]
        return img
            
      
class Encoder_Decoder():
    def __init__(self,list_patches,path_model,path_model2, path_destino, batch_size,op_bit_allocation,cuda,qiters,
                 target, height,width,h,w,min_iters,color_mode):
      
        self.list_patches=list_patches
        self.path_model = path_model
        self.path_model2 = path_model2
        self.path_destino = path_destino
        self.batch_size=batch_size
        self.op_bit_allocation=op_bit_allocation
        self.cuda = cuda
        self.qiters = qiters
        self.target = target
        self.height=height
        self.width=width
        self.h=h
        self.w =w
        self.min_iters = min_iters
        self.color_mode = color_mode
    
    
    def ed_process(self,my_object):
      
        
        bpp=[]
        #for th in self.threshold:
        #for q in range(self.qiters, self.qiters+1):
        th =self.target
        train_loader = torch.utils.data.DataLoader(self.list_patches,batch_size = self.batch_size,num_workers=2,
                                                   shuffle=False)
        num_batch = len(train_loader)

        encoder = network.EncoderCell(1)
        binarizer = network.Binarizer(28)
        decoder = network.DecoderCell(28,1)

        encoder2 = network.EncoderCell(2)
        binarizer2 = network.Binarizer(4)
        decoder2 = network.DecoderCell(4,2)

        encoder.load_state_dict(torch.load(self.path_model,map_location=lambda storage, loc: storage ) )
        binarizer.load_state_dict(torch.load(self.path_model.replace('encoder', 'binarizer'),map_location=lambda storage, loc: storage ))
        decoder.load_state_dict(torch.load(self.path_model.replace('encoder', 'decoder'),map_location=lambda storage, loc: storage))

        encoder2.load_state_dict(torch.load(self.path_model2,map_location=lambda storage, loc: storage ) )
        binarizer2.load_state_dict(torch.load(self.path_model2.replace('encoder', 'binarizer'),map_location=lambda storage, loc: storage ))
        decoder2.load_state_dict(torch.load(self.path_model2.replace('encoder', 'decoder'),map_location=lambda storage, loc: storage))

        if cuda:
            encoder = encoder.cuda()
            binarizer = binarizer.cuda()
            decoder = decoder.cuda()
            encoder2 = encoder2.cuda()
            binarizer2 = binarizer2.cuda()
            decoder2 = decoder2.cuda()
        qbits=np.zeros(1)

        list_patches_recons = np.zeros((self.batch_size*num_batch,self.height,self.width,3),dtype='uint8')

        print('Batch size',self.batch_size,'Número de batches: ',num_batch,', Dimensão patch',self.h,'x',self.w)
        code_img=[]
        code_img2 =[]
        for batch,data in enumerate(train_loader):
            image = data 
            if cuda:
              image = image.cuda()
            codes_final=[]
            #assert height % 32 == 0 and width % 32 == 0
            with torch.no_grad():
                image = Variable(image)
           
                encoder_h_1 = ((torch.zeros(self.batch_size, 256, self.height // 4, self.width // 4)),
                                  (torch.zeros(self.batch_size, 256, self.height // 4, self.width // 4)))
                encoder_h_2 = ((torch.zeros(self.batch_size, 512, self.height // 8, self.width // 8)),
                                     (torch.zeros(self.batch_size, 512, self.height // 8, self.width // 8)))
                encoder_h_3 = ((torch.zeros(self.batch_size, 512, self.height // 16, self.width // 16)),
                                     (torch.zeros(self.batch_size, 512, self.height // 16, self.width // 16)))
                decoder_h_1 = ((torch.zeros(self.batch_size, 512, self.height // 16, self.width // 16)),
                                     (torch.zeros(self.batch_size, 512, self.height // 16, self.width // 16)))
                decoder_h_2 = ((torch.zeros(self.batch_size, 512, self.height // 8, self.width // 8)),
                                     (torch.zeros(self.batch_size, 512, self.height // 8, self.width // 8)))
                decoder_h_3 = ((torch.zeros(self.batch_size, 256, self.height // 4, self.width // 4)),
                                     (torch.zeros(self.batch_size, 256, self.height // 4, self.width // 4)))
                decoder_h_4 = ((torch.zeros(self.batch_size, 128, self.height // 2, self.width // 2)),
                                     (torch.zeros(self.batch_size, 128, self.height // 2, self.width // 2)))

            encoder_h_1 =  (encoder_h_1[0], encoder_h_1[1])
            encoder_h_2 =  (encoder_h_2[0], encoder_h_2[1])
            encoder_h_3 =  (encoder_h_3[0], encoder_h_3[1])

            decoder_h_1 =  (decoder_h_1[0], decoder_h_1[1])
            decoder_h_2 =  (decoder_h_2[0], decoder_h_2[1])
            decoder_h_3 =  (decoder_h_3[0], decoder_h_3[1])
            decoder_h_4 =  (decoder_h_4[0], decoder_h_4[1])  

            encoder_h_1_2 = (encoder_h_1[0], encoder_h_1[1])
            encoder_h_2_2 = (encoder_h_2[0], encoder_h_2[1])
            encoder_h_3_2 = (encoder_h_3[0], encoder_h_3[1])

            decoder_h_1_2 = (decoder_h_1[0], decoder_h_1[1])
            decoder_h_2_2 = (decoder_h_2[0], decoder_h_2[1])
            decoder_h_3_2 = (decoder_h_3[0], decoder_h_3[1])
            decoder_h_4_2 = (decoder_h_4[0], decoder_h_4[1])  

            if cuda:
                encoder_h_1 = (encoder_h_1[0].cuda(), encoder_h_1[1].cuda())
                encoder_h_2 = (encoder_h_2[0].cuda(), encoder_h_2[1].cuda())
                encoder_h_3 = (encoder_h_3[0].cuda(), encoder_h_3[1].cuda())

                decoder_h_1 = (decoder_h_1[0].cuda(), decoder_h_1[1].cuda())
                decoder_h_2 = (decoder_h_2[0].cuda(), decoder_h_2[1].cuda())
                decoder_h_3 = (decoder_h_3[0].cuda(), decoder_h_3[1].cuda())
                decoder_h_4 = (decoder_h_4[0].cuda(), decoder_h_4[1].cuda())    

                encoder_h_1_2 = (encoder_h_1[0].cuda(), encoder_h_1[1].cuda())
                encoder_h_2_2 = (encoder_h_2[0].cuda(), encoder_h_2[1].cuda())
                encoder_h_3_2 = (encoder_h_3[0].cuda(), encoder_h_3[1].cuda())

                decoder_h_1_2 = (decoder_h_1[0].cuda(), decoder_h_1[1].cuda())
                decoder_h_2_2 = (decoder_h_2[0].cuda(), decoder_h_2[1].cuda())
                decoder_h_3_2 = (decoder_h_3[0].cuda(), decoder_h_3[1].cuda())
                decoder_h_4_2 = (decoder_h_4[0].cuda(), decoder_h_4[1].cuda())    

            image = image - 0.5
            image = image.float() 

            im  = torch.zeros(batch_size, 1, self.height, self.width) + 0.5
            im2 = torch.zeros(batch_size, 2, self.height, self.width) + 0.5

            res = torch.zeros(batch_size, 1, self.height, self.width,dtype=torch.float).cuda()
    
            res[:,0,:,:] = image[:,0,:,:]
            res2 = image[:,1:,:,:]

            result = 0
            codes  = []
            codes2 = []

            for iters in range(self.qiters):

                encoded, encoder_h_1, encoder_h_2, encoder_h_3 = encoder(res, encoder_h_1, encoder_h_2, encoder_h_3)
                code = binarizer(encoded)
                output, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4 = decoder(code, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4)

                encoded2, encoder_h_1_2, encoder_h_2_2, encoder_h_3_2 = encoder2(res2, encoder_h_1_2, encoder_h_2_2, encoder_h_3_2)
                code2 = binarizer2(encoded2)
                output2, decoder_h_1_2, decoder_h_2_2, decoder_h_3_2, decoder_h_4_2 = decoder2(code2, decoder_h_1_2, decoder_h_2_2, decoder_h_3_2, decoder_h_4_2)

                im += output.data.cpu()
                res = res - output
                codes.append(code.data.cpu().numpy())

                im2 += output2.data.cpu()
                res2 = res2 - output2
                codes2.append(code2.data.cpu().numpy())
                
                qbits+= (self.height/16*self.width/16 * 32)*self.batch_size 
                #print('code.shape',code.shape,'code2.shape', code2.shape)
                for i in range(self.batch_size):
                    
                    im_new = torch.cat((im, im2), 1)
                    #print(im_new.shape)
                    img = np.squeeze(im_new[i].numpy().clip(0, 1) * 255.0).astype(np.uint8).transpose(1, 2, 0)
                    list_patches_recons[i+batch*self.batch_size,:,:,:] = img
                
                if self.op_bit_allocation and iters+1>=self.min_iters:
                    result =[]
                    for i in range(self.batch_size):
                        patch_test = np.squeeze(im[i].numpy().clip(0, 1) * 255.0).astype(np.uint8).transpose(1, 2, 0)
                        #patch_test=np.squeeze(im.numpy().clip(0, 1) * 255.0).astype(np.uint8).transpose(1, 2, 0)
                        patch_test = Image.fromarray(patch_test, mode = self.color_mode)
                        patch_test = patch_test.convert('RGB')

                        patch_ref = (patches[batch*batch_size+i]* 255.0).astype(np.uint8)
                        #patch_ref  = Image.fromarray(patch_ref, mode='RGB')
                        patch_ref = Image.fromarray(patch_ref, mode = self.color_mode)
                        patch_ref = patch_ref.convert('RGB')

                        result.append(calc_metric(patch_ref,patch_test,'psnr')) 
                    lower = min(result)
                    media = np.mean(result)
                    if media>=th and lower>=th*0.90:
                        print('iteração',iters+1,'PSNR médio', media)
                        break

            bpp   = qbits/(768*512)   
             
            codes1  = np.asarray(codes).flatten()
            codes2  =  np.asarray(codes2).flatten()

            codes12 = np.concatenate((codes1, codes2), axis=0)
            if batch==0:
                code_img = codes12
                code_img1  = codes1
                code_img2 = codes2
            else: 
                code_img = np.concatenate((code_img, codes12), axis=0)                                
                code_img1 = np.concatenate((code_img1, codes1), axis=0)                                
                code_img2 = np.concatenate((code_img2, codes2), axis=0)                                

                 
        
        code_img1 = list(map(int, code_img1))
        latent1 = list(map(lambda x: x != -1, code_img1))
        size_in1 = bitstring.Bits(latent1).length
        compressed1 = gzip.compress(bitstring.Bits(latent1).bytes)
        size_out1 = bitstring.Bits(compressed1).length 
        #print('entrada (bits)', size_in,'saída(bits)', size_out, 'Data compression ratio',size_in/size_out)
        #bpp2 = size_out/(512*768)

        code_img2 = list(map(int, code_img2))
        latent2 = list(map(lambda x: x != -1, code_img2))
        size_in2 = bitstring.Bits(latent2).length
        compressed2 = gzip.compress(bitstring.Bits(latent2).bytes)
        size_out2 = bitstring.Bits(compressed2).length 
        print('entrada (bits)', size_in1+size_in1,'saída(bits)', size_out1+size_out2, 'Data compression ratio',(size_in1+size_in2)/(size_out1 + size_out2))
        bpp2 = (size_out1+size_out2)/(512*768)
        return list_patches_recons,bpp,bpp2


                        
          
    
    def resultados(self,list_patches_recons,img_original,my_object,path_save,color_mode, type_psnr='RGB'):

        array_patches_recons = np.asarray(list_patches_recons)
        x = my_object.reconstruct_image(array_patches_recons,self.h,self.w)

        img_obtida   = Image.fromarray(x, mode=color_mode)              

        img_original = Image.fromarray(img_original, mode=color_mode)
        
        img_original = img_original.convert('RGB')
        img_obtida   = img_obtida.convert('RGB')      
        
        msssim = 0  

        if type_psnr == 'Y':
            img_original = img_original.convert('YCbCr')
            img_obtida = img_obtida.convert('YCbCr')
            img_obtida, _, _ = img_obtida.split()
            img_original, _, _ = img_original.split()


        psnr   = my_object.calc_metric(img_original, img_obtida,'psnr')
        ssim   = my_object.calc_metric(img_original, img_obtida,'ssim')
        if type_psnr == 'RGB':
            msssim = my_object.calc_metric(img_original, img_obtida,'msssim')  

        return psnr, ssim, msssim



def calc_mean(values, name):
  # l,c = values.shape
  
  # for y in range(c):
  #     ls=[]
  #     for x in range(l):
  #         ls.append(values[x][y])
  print(name, str(np.mean(values)).replace('.',','))  
           


my_object = Manipulation_Images()
filenames = glob.glob(path_load)
j=-1
psnr = np.zeros(len(filenames))
ssim = np.zeros(len(filenames))
msssim = np.zeros(len(filenames))
bpp = np.zeros(len(filenames))
bpp2 = np.zeros(len(filenames))

for filename in filenames: 
    j+=1 
    print(filename)
    img_original = my_object.load_image(filename, colorspace_input)
    w,h,c= img_original.shape
    patches = my_object.extract_img_patch(img_original,size_patch) 
    patches = patches/255.0 
    
    patches_transpose = np.transpose(patches, (0,3,1,2))
    list_patches      = torch.from_numpy(patches_transpose)
    
    my_object_ed = Encoder_Decoder(list_patches,path_model,path_model2,path_destino, batch_size,op_bit_allocation, cuda,
                                   qiters,target_psnr,height,width,h,w,min_iters,colorspace_input)
    
    list_patches_recons, bpp[j],bpp2[j]  = my_object_ed.ed_process(my_object)
    psnr[j], ssim[j], msssim[j] = my_object_ed.resultados(list_patches_recons,img_original,my_object,path_save,colorspace_input,type_psnr)
 
    print('Target PSNR %.2f dB. Output: BPP %.4f, PSNR %.4f dB, SSIM %.4f, MS-SSIM %.4f'% (target_psnr,bpp[j], psnr[j], ssim[j],msssim[j]))
        
calc_mean(psnr, 'psnr')
calc_mean(ssim, 'ssim')
calc_mean(msssim, 'ms-ssim')
calc_mean(bpp, 'bpp nominal')
calc_mean(bpp2, 'bpp after entropy coding')
