import time
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
import network
import PIL
import sys
from PIL import Image
from pathlib import PosixPath
import pickle
import network
from skimage.util import view_as_blocks
import matplotlib.pyplot as plt
from img_common.msssim import compare_msssim
import network
from skimage.measure import compare_psnr, compare_ssim
import glob



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

            
    def load_image(self,path, data_format=None, color_mode='RGB'):

        valid, img = self.is_pillow_valid_img(path)
        if not valid:
            return []

        if color_mode:
            img = img.convert(color_mode)
        img_data = np.array(img)
        img.close()

        if data_format:
            img_data = self.conv_data_format(img_data, data_format)

        return img_data

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
    def __init__(self,list_patches,path_model,path_destino, batch_size,op_bit_allocation,cuda,qiters,
                 threshold, height,width,h,w):

        self.list_patches=list_patches
        self.path_model = path_model
        self.path_destino = path_destino
        self.batch_size=batch_size
        self.op_bit_allocation=op_bit_allocation
        self.cuda = cuda
        self.qiters = qiters
        self.threshold = threshold
        self.height=height
        self.width=width
        self.h=h
        self.w =w
    
    
    def ed_process(self,my_object):
        bpp=[]
        #for th in self.threshold:
        for q in range(self.qiters, self.qiters+1):
            train_loader = torch.utils.data.DataLoader(self.list_patches,batch_size = self.batch_size,num_workers=2,
                                                       shuffle=False)
            num_batch = len(train_loader)
            encoder = network.EncoderCell()
            binarizer = network.Binarizer()
            decoder = network.DecoderCell()

            encoder.load_state_dict(torch.load(self.path_model,map_location=lambda storage, loc: storage ) )
            binarizer.load_state_dict(torch.load(self.path_model.replace('encoder', 'binarizer'),map_location=lambda storage, loc: storage ))
            decoder.load_state_dict(torch.load(self.path_model.replace('encoder', 'decoder'),map_location=lambda storage, loc: storage))

            encoder = encoder.cuda()
            binarizer = binarizer.cuda()
            decoder = decoder.cuda()

            qbits=np.zeros((self.qiters),dtype='float64')
            im=0
            list_patches_recons = np.zeros((self.qiters,self.batch_size*num_batch,32,32,3),dtype='uint8' )
         

            for batch,data in enumerate(train_loader):
                image = data.cuda()
                codes_final=[]
                assert height % 32 == 0 and width % 32 == 0
                with torch.no_grad():
                    image = Variable(image)
                res   = image - 0.5
                codes = []

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

                encoder_h_1 = (encoder_h_1[0].cuda(), encoder_h_1[1].cuda())
                encoder_h_2 = (encoder_h_2[0].cuda(), encoder_h_2[1].cuda())
                encoder_h_3 = (encoder_h_3[0].cuda(), encoder_h_3[1].cuda())

                decoder_h_1 = (decoder_h_1[0].cuda(), decoder_h_1[1].cuda())
                decoder_h_2 = (decoder_h_2[0].cuda(), decoder_h_2[1].cuda())
                decoder_h_3 = (decoder_h_3[0].cuda(), decoder_h_3[1].cuda())
                decoder_h_4 = (decoder_h_4[0].cuda(), decoder_h_4[1].cuda())    

                im = torch.zeros(batch_size, 3, 32, 32) + 0.5
                res=res.float()
                result=0

                for iters in range(self.qiters):

                    encoded, encoder_h_1, encoder_h_2, encoder_h_3 = encoder(res, encoder_h_1, encoder_h_2, encoder_h_3)
                    code = binarizer(encoded)
                    output, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4 = decoder(code, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4)
                    im+=output.data.cpu()
                    res = res - output
                    codes.append(code.data.cpu().numpy())
                    codes_final.append(codes)
                    
                    for i in range(self.batch_size):
                        img = np.squeeze(im[i].numpy().clip(0, 1) * 255.0).astype(np.uint8).transpose(1, 2, 0)
                        list_patches_recons[iters][i+batch*self.batch_size] = img
                        
               
                    if self.op_bit_allocation:
                        for i in range(self.batch_size):
                            patch_test = np.squeeze(im[i].numpy().clip(0, 1) * 255.0).astype(np.uint8).transpose(1, 2, 0)
                            #patch_test=np.squeeze(im.numpy().clip(0, 1) * 255.0).astype(np.uint8).transpose(1, 2, 0)
                            patch_test = Image.fromarray(patch_test, mode='RGB')
                            patch_ref  =  (patches[batch*batch_size+i]* 255.0).astype(np.uint8)
                            patch_ref = Image.fromarray(patch_ref, mode='RGB')
                            result    =   calc_psnr(patch_ref,patch_test) 
                            result= result/batch_size    
                        if result>th:
                            break
                    qbits[iters] += 128*(iters+1)*self.batch_size 
    
    
                
                bpp = qbits/(768*512)                         
            
                        
        return  list_patches_recons,bpp                     
          
    
    def resultados(self,list_patches_recons,img_original,my_object):
        psnr   = np.zeros((self.qiters), dtype='float64')
        ssim   = np.zeros((self.qiters), dtype='float64')
        msssim = np.zeros((self.qiters), dtype='float64')


        for j in range(self.qiters):
        
            array_patches_recons = np.asarray(list_patches_recons[j])
            x = my_object.reconstruct_image(array_patches_recons,self.h,self.w)
            img_obtida = Image.fromarray(x, mode='RGB')
            
      
            psnr[j]   = my_object.calc_metric(img_original, img_obtida,'psnr')
            ssim[j]   = my_object.calc_metric(img_original, img_obtida,'ssim')
            msssim[j] = my_object.calc_metric(img_original, img_obtida,'msssim')  
        return psnr, ssim, msssim

          
def calc_mean(values, name):
    l,c = values.shape
    print(name)
    values_mean=[]
    for y in range(c):
        ls=[]
        for x in range(l):
            ls.append(values[x][y])
        values_mean.append(str(np.mean(ls)).replace('.',','))
        print(str(np.mean(ls)).replace('.',','))  
    np.save(name,values_mean)  
      
path='imagens_teste/'
path_model='checkpoint/cp4_mse/encoder_epoch_2.pth'
path_destino = 'resultados'
input_channels = 3
height=32 
width =32
cuda =1
op_bit_allocation=False
batch_size=64
qiters=16
threshold = np.arange(16, 41, 0.5)
my_object = Manipulation_Images()
filenames=glob.glob(path+'/*2.bmp*')

psnr   = np.zeros((len(filenames), qiters) ,dtype='float64')
ssim   = np.zeros((len(filenames), qiters) ,dtype='float64')
msssim = np.zeros((len(filenames), qiters) ,dtype='float64')
bpp = np.zeros((len(filenames), qiters) ,dtype='float64')
j=-1
for filename in filenames: 
    j+=1 
    print(filename)
    img_original = my_object.load_image(filename)
    #print((img_original))

    w,h,c= img_original.shape
    patches = my_object.extract_img_patch(img_original,32) 
    patches = patches/255.0 
    patches_transpose = np.transpose(patches, (0,3,1,2))
    list_patches = torch.from_numpy(patches_transpose)
    

    my_object_ed = Encoder_Decoder(list_patches,path_model,path_destino, batch_size,op_bit_allocation, cuda,
                                   qiters,threshold,height,width,h,w)
    
    list_patches_recons,bpp[j][:]  = my_object_ed.ed_process(my_object)
    psnr[j][:], ssim[j][:], msssim[j][:] = my_object_ed.resultados(list_patches_recons,img_original,my_object)
 
calc_mean(psnr,'psnr')
calc_mean(ssim,'ssim')
calc_mean(msssim,'ms-ssim')
calc_mean(bpp,'bpp')

