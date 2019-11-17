num_img = '*'
epoch = '23'
path_save = 'resultados/bit_allocation/epoch'+epoch+'/'

path_load = 'imagens_teste/kodim'+str(num_img)+'.bmp'
#path_model  = '/media/data/Datasets/samsung/modelos/rnn/adam_mse_l1_beta1/encoder_epoch_1.pth'
#path_model = './checkpoint/mse_l1_ds_Marcelo_lambda_-0.07Nivel_28niveis_latente2_2_46/encoder_epoch_'+epoch+'.pth'
path_model = './checkpoint/mse_l1_ds_Marcelo_lambda_-0.01Nivel_28niveis/encoder_epoch_'+epoch+'.pth'
#path_model ='./checkpoint/mse_l1_ds_Marcelo_lambda_6_10-7_exp-0.07Nivel_28niveis_latente2_2_46/encoder_epoch_'+epoch+'.pth'

op_save = 0
op_bit_allocation = 1

offset = 0.5
input_channels = 3
size_patch = 32

batch_size = 4
target_psnr = range(25,43)

num_min_iter = 5
num_max_iter = 24

canais = 32
colorspace_input = 'RGB'
height = size_patch
width  = size_patch
cuda = 1

import time, bitstring
import os
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import dataset
import PIL
import sys
import pandas as pd
from PIL import Image
from pathlib import PosixPath
import pickle
import network
from skimage.util import view_as_blocks
from img_common.msssim import compare_msssim
from skimage.measure import compare_psnr, compare_ssim
import glob,gzip
torch.cuda.get_device_name(1)
CUDA_VISIBLE_DEVICES=1
torch.cuda.set_device(1)


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
        #print(img)
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
    def __init__(self,list_patches,path_model, batch_size,op_bit_allocation,cuda,num_max_iter,target, height,width,h,w,num_min_iter,color_mode):
        self.list_patches =list_patches
        self.path_model = path_model
        self.batch_size = batch_size
        self.op_bit_allocation = op_bit_allocation
        self.cuda = cuda
        self.num_max_iter = num_max_iter
        self.target = target
        self.height = height
        self.width = width
        self.h = h
        self.w = w
        self.num_min_iter = num_min_iter
        self.color_mode = color_mode
      
    def ed_process(self,my_object):
        bpp=[]
        code_img = []
        th =self.target
        train_loader = torch.utils.data.DataLoader(self.list_patches,batch_size = self.batch_size,num_workers=2,
                                                   shuffle=False)
        num_batch = len(train_loader)
        encoder = network.EncoderCell(3).eval()
        binarizer = network.Binarizer(canais).eval()
        decoder = network.DecoderCell(canais,3).eval()

        encoder.load_state_dict(torch.load(self.path_model,map_location=lambda storage, loc: storage ) )
        binarizer.load_state_dict(torch.load(self.path_model.replace('encoder', 'binarizer'),map_location=lambda storage, loc: storage ))
        decoder.load_state_dict(torch.load(self.path_model.replace('encoder', 'decoder'),map_location=lambda storage, loc: storage))

        if cuda:
            encoder = encoder.cuda()
            binarizer = binarizer.cuda()
            decoder = decoder.cuda()

        qbits = np.zeros(1)
        list_patches_recons =  np.zeros((self.batch_size*num_batch,self.height,self.width,3),dtype='uint8')
        
        for batch,data in enumerate(train_loader):
            image = data 
            if cuda:
              image = image.cuda()
            
            #assert height % 32 == 0 and width % 32 == 0
          
            with torch.no_grad():  
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

            encoder_h_1 = (encoder_h_1[0], encoder_h_1[1])
            encoder_h_2 = (encoder_h_2[0], encoder_h_2[1])
            encoder_h_3 = (encoder_h_3[0], encoder_h_3[1])

            decoder_h_1 = (decoder_h_1[0], decoder_h_1[1])
            decoder_h_2 = (decoder_h_2[0], decoder_h_2[1])
            decoder_h_3 = (decoder_h_3[0], decoder_h_3[1])
            decoder_h_4 = (decoder_h_4[0], decoder_h_4[1])  

            if cuda:
                encoder_h_1 = (encoder_h_1[0].cuda(), encoder_h_1[1].cuda())
                encoder_h_2 = (encoder_h_2[0].cuda(), encoder_h_2[1].cuda())
                encoder_h_3 = (encoder_h_3[0].cuda(), encoder_h_3[1].cuda())

                decoder_h_1 = (decoder_h_1[0].cuda(), decoder_h_1[1].cuda())
                decoder_h_2 = (decoder_h_2[0].cuda(), decoder_h_2[1].cuda())
                decoder_h_3 = (decoder_h_3[0].cuda(), decoder_h_3[1].cuda())
                decoder_h_4 = (decoder_h_4[0].cuda(), decoder_h_4[1].cuda())    

     
           
            im      = torch.zeros(batch_size, 3, self.height, self.width)
            res     = (image - offset).float()
            codes   = []
 
            for iters in range(self.num_max_iter):
                  
                encoded, encoder_h_1, encoder_h_2, encoder_h_3 = encoder(res, encoder_h_1, encoder_h_2, encoder_h_3)
                code = binarizer(encoded)
                output, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4 = decoder(code, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4)
                im += output.data.cpu()
                res = res - output
                codes.append(code.data.cpu().numpy())
                qbits += (self.height/16*self.width/16 *canais)*self.batch_size 
  
                if self.op_bit_allocation and iters+1>=self.num_min_iter:
                    result =[]
                    for i in range(self.batch_size):
                        patch = im[i]+offset 
                        patch = np.squeeze(patch.numpy().clip(0, 1) * 255.0).astype(np.uint8).transpose(1, 2, 0)
                        list_patches_recons[i+batch*self.batch_size,:,:,:] = patch

                        patch_test = patch
                        patch_test = Image.fromarray(patch_test, mode = self.color_mode)
                        patch_test = patch_test.convert('RGB')

                        patch_ref = (patches[batch*batch_size+i]* 255.0).astype(np.uint8)
                        patch_ref = Image.fromarray(patch_ref, mode = self.color_mode)
                        patch_ref = patch_ref.convert('RGB')

                        result.append(calc_metric(patch_ref,patch_test,'psnr')) 
                    lower = min(result)
                    media = np.mean(result)
                    if media>=th and lower>=th*0.95:
                        print('iteração',iters+1,'PSNR média', media)
                        break

            if not self.op_bit_allocation:
                for i in range(self.batch_size):
                    patch = im[i] + offset 
                    patch = np.squeeze(patch.numpy().clip(0, 1) * 255.0).astype(np.uint8).transpose(1, 2, 0)
                    list_patches_recons[i+batch*self.batch_size,:,:,:] = patch

            codes = np.asarray(codes).flatten()
            if batch==0:
                code_img = codes
            else: 
                code_img = np.concatenate((code_img, codes), axis=0)
            codes=[]

        code_img = list(map(int, code_img))
        latent = list(map(lambda x: x != -1, code_img))
        size_in = bitstring.Bits(latent).length
        compressed = gzip.compress(bitstring.Bits(latent).bytes)
        size_out = bitstring.Bits(compressed).length 
                
        bpp2 = size_out/(512*768)            
        bpp = qbits/(768*512)          
        return list_patches_recons,bpp,bpp2                     
          
    
    def resultados(self,list_patches_recons,img_original,my_object,path_save,color_mode,name_img,op_save):

        array_patches_recons = np.asarray(list_patches_recons)
        x = my_object.reconstruct_image(array_patches_recons,self.h,self.w)

        img_obtida   = Image.fromarray(x, mode=color_mode)              
        img_original = Image.fromarray(img_original, mode=color_mode)
        
        img_original = img_original.convert('RGB')
        img_obtida   = img_obtida.convert('RGB')      

        if op_save:
            img_obtida.save(path_save+name_img+'.png')

        psnr   = my_object.calc_metric(img_original, img_obtida,'psnr')
        ssim   = my_object.calc_metric(img_original, img_obtida,'ssim')
        msssim = my_object.calc_metric(img_original, img_obtida,'msssim')  

        img_original = img_original.convert('YCbCr')
        img_obtida = img_obtida.convert('YCbCr')
        img_obtida, _, _ = img_obtida.split()
        img_original, _, _ = img_original.split()
        psnr_y   = my_object.calc_metric(img_original, img_obtida,'psnr')
       
        return psnr,psnr_y,  ssim, msssim


def calc_mean(values):
    return np.mean(values)

def print_resultados(values):
    for v in values:
        print(v)


my_object = Manipulation_Images()
filenames = glob.glob(path_load)

psnr_iter = []
psnr_y_iter = []
ssim_iter = []
msssim_iter = []
bpp_iter = []
bpp2_iter = []
g_iter = []

            
if not os.path.exists(path_save):
    os.mkdir(path_save)

for t_psnr in target_psnr:
    j=-1
    psnr = np.zeros(len(filenames))
    psnr_y = np.zeros(len(filenames))
    ssim = np.zeros(len(filenames))
    msssim = np.zeros(len(filenames))
    bpp = np.zeros(len(filenames))
    bpp2 = np.zeros(len(filenames))
    g = np.zeros(len(filenames))
   
    
    for filename in filenames: 
        j+=1 
        name_img = (filename.split('/')[1]).split('.')[0]

        try:
            df = pd.read_csv(path_save+name_img+'_epoch'+str(epoch)+'.csv')
            print('open')
        except:
            print('new df')
            dados = {'name_img':[] ,'target_psnr':[], 'bpp_nominal':[], 'bpp_real':[], 'psnr':[], 'psnr_y':[], 'ssim':[], 'ms-ssim':[],'ganho_bpp':[]}
            df = pd.DataFrame(dados,columns=['name_img','target_psnr','bpp_nominal','bpp_real', 'psnr','psnr_y','ssim','ms-ssim','ganho_bpp'],dtype=float)


        img_original = my_object.load_image(filename, colorspace_input)
        w,h,c= img_original.shape
        patches = my_object.extract_img_patch(img_original,size_patch) 
        patches = patches/255.0 
        
        patches_transpose = np.transpose(patches, (0,3,1,2))
        list_patches      = torch.from_numpy(patches_transpose)
        

        my_object_ed = Encoder_Decoder(list_patches,path_model, batch_size,op_bit_allocation, cuda,
                                       num_max_iter,t_psnr,height,width,h,w,num_min_iter,colorspace_input)
        
        list_patches_recons, bpp[j],bpp2[j]   = my_object_ed.ed_process(my_object)
        psnr[j],psnr_y[j], ssim[j], msssim[j] = my_object_ed.resultados(list_patches_recons,img_original,my_object,path_save,colorspace_input,name_img,op_save)         
        
        print('Image: %s, Target PSNR %.2f dB. BPP nominal: %.4f, BPP entropy code: %.4f, PSNR (RGB): %.4fdB, PSNR Y: %.4f dB, SSIM: %.4f, MS-SSIM: %.4f'% (name_img,t_psnr,bpp[j],bpp2[j], psnr[j], psnr_y[j], ssim[j],msssim[j]))
        g[j] = (bpp[j]-bpp2[j])*100/bpp[j]
        df= df.append(pd.Series([name_img, t_psnr, float(bpp[j]),bpp2[j],psnr[j],psnr_y[j],ssim[j],msssim[j],g[j]], index=df.columns), ignore_index=True)
        df.to_csv(path_save+name_img+'_epoch'+str(epoch)+'.csv',index=False)

    psnr_iter.append(calc_mean(psnr))
    psnr_y_iter.append(calc_mean(psnr_y))
    ssim_iter.append(calc_mean(ssim))
    msssim_iter.append(calc_mean(msssim))
    bpp_iter.append(calc_mean(bpp))
    bpp2_iter.append(calc_mean(bpp2))
    g_iter.append(calc_mean(g))


print('PSNR')
print_resultados(psnr_iter)
print('PSNR Y')
print_resultados(psnr_y_iter)
print('SSIM')
print_resultados(ssim_iter)
print('MS-SSIM')
print_resultados(msssim_iter)
print('BPP')
print_resultados(bpp_iter)
print('BPP Real')
print_resultados(bpp2_iter)

try:
    df = pd.read_csv(path_save+'media_por_target_epoch'+str(epoch)+'.csv')
except:
   dados = {'bpp_nominal':[], 'bpp_real':[], 'psnr':[], 'psnr_y':[], 'ssim':[], 'ms-ssim':[],'ganho_bpp':[]}
   df = pd.DataFrame(dados,columns=['bpp_nominal','bpp_real', 'psnr','psnr_y','ssim','ms-ssim','ganho_bpp'],dtype=float)


#dados = {'bpp_nominal':np.asarray(bpp_iter), 'bpp_real':np.asarray(bpp2_iter), 'psnr':np.asarray(psnr_iter), 'psnr_y':np.asarray(psnr_y_iter), 'ssim':np.asarray(ssim_iter), 'ms-ssim':np.asarray(msssim_iter),'ganho_bpp':np.asarray(g_iter)}
for j in range(len(bpp_iter)):
    df= df.append(pd.Series([np.asarray(bpp_iter[j]),np.asarray(bpp2_iter[j]),np.asarray(psnr_iter[j]),np.asarray(psnr_y_iter[j]), np.asarray(ssim_iter[j]),np.asarray(msssim_iter[j]),np.asarray(g_iter[j])], index=df.columns), ignore_index=True)
#df = pd.DataFrame(dados,columns=['bpp_nominal','bpp_real', 'psnr','psnr_y','ssim','ms-ssim','ganho_bpp'],dtype=float)
df.to_csv(path_save+'media_por_target_epoch'+str(epoch)+'.csv',index=False)
