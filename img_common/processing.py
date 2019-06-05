""" This file has operations for image processing """


import numpy as np
import gzip
from enums import Metrics
from io import BytesIO
from PIL import Image
from skimage import img_as_ubyte, img_as_float
from skimage.measure import compare_psnr, compare_ssim
from skimage.util import view_as_blocks
from bitstring import Bits
from pathlib import PosixPath
from msssim import compare_msssim
import pandas as pd
import os
from pathlib import Path
import warnings

from enums import ImgData


class ImgProc:
    """ Static class that has methods to make basic operations on images """
    @staticmethod
    def pad_img(img, patch_size):
        """ Method that receives an image and a size for the patches. The method
            pad the image so that they can be cropped later
        """
        orig_shape = np.array(img.shape[:2])
        new_shape = patch_size * np.ceil(orig_shape / patch_size).astype(int)
        points_to_pad = new_shape - orig_shape
        pad_img = np.pad(img, [(0, points_to_pad[0]), (0, points_to_pad[1]),
                         (0, 0)], 'edge')
        return pad_img

    @staticmethod
    def is_pillow_valid_img(pathname, return_ref=True):
        """ Function that verifies if the file is a valid image considering
            the pillow library, that's used in this code. If desired, it
            returns the opened ref. The retuned reference must be closed later.
        """
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

    @staticmethod
    def get_img_real_ext(pathname):
        """ Function that identifies the format of the image and return it as
            a lower case string. If it's not a valid image, returns None.
        """
        valid, img = ImgProc.is_pillow_valid_img(pathname)
        if not valid:
            return None

        format = img.format.lower()
        img.close()
        return format

    @staticmethod
    def reconstruct_image(patches, width, height):
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

    @staticmethod
    # TODO: use view_as_windows to implement extraction of overlapping patches
    def extract_img_patch(orig_img, patch_size):
        """ Method that receives an image and the patch size and extract
            the patches of the image.
        """
        if np.all(np.equal(orig_img.shape, patch_size)):
            return orig_img

        img = ImgProc.pad_img(orig_img, patch_size)
        color = 1
        if len(img.shape) > 2:
            color = img.shape[2]
        patches = view_as_blocks(img, (patch_size, patch_size, color))
        patches = patches.reshape(-1, patch_size, patch_size, color)
        return patches

    @staticmethod
    def conv_data_format(img, data_format):
        """ Method that receives a valid image array and a desired format to
            convert into.
        """
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

    @staticmethod
    def save_img_from_ref(img_ref, new_path, kwargs={}):
        """ Utils function to rename a image. It's useful for conversion of
            images. It's also useful as a wrapper to avoid importing the PIL in
            other files
        """
        if isinstance(img_ref, (PosixPath, str)):
            img_ref = Image.open(img_ref)
        if isinstance(img_ref, np.ndarray):
            img_ref = Image.fromarray(img_ref)

        img_ref.save(new_path, **kwargs)

    @staticmethod
    def load_image(path, data_format=None, color_mode='RGB'):
        """ This method receives an image pathname and the target colorspace.
            If the path points to a non valid image, it returns empty data.
        """
        valid, img = ImgProc.is_pillow_valid_img(path)
        if not valid:
            return []

        if color_mode:
            img = img.convert(color_mode)
        img_data = np.array(img)
        img.close()

        if data_format:
            img_data = ImgProc.conv_data_format(img_data, data_format)

        return img_data

    @staticmethod
    def calc_n_patches(info, patch_size):
        """ Calculates the number of patches images with this height and
            widht considering the padding """
        if isinstance(info, (PosixPath, str)):
            img = Image.open(info)
            width, height = img.size
        elif isinstance(info, list):
            width, height = info
        else:
            width, height = info

        line_patches = np.ceil(height / patch_size).astype(int)
        column_patches = np.ceil(width / patch_size).astype(int)
        num_of_patches = line_patches * column_patches
        return num_of_patches

    @staticmethod
    def calc_bpp_using_gzip(list_arrays, pixels_num, bpp_proxy, pos):
        """ Function that calculates the bpp considering the gzip. It receives
            a data_array representing an image.
        """
        try:
            # If the number is not already an integer, it's not been quantized
            # So it's not fair do estimate bpp using a round version
            if np.all(np.equal(np.mod(list_arrays[0], 1), 0)):
                list_arrays = list(map(lambda e: np.array(e).astype(np.int),
                                       list_arrays))

            compressed = list(map(gzip.compress, list_arrays))
            # Bits has many representations. length get the len in bits
            bpp = list(map(lambda c: Bits(c).length / pixels_num, compressed))
            bpp_proxy[pos] = list(np.cumsum(bpp))
        except Exception as e:
            print(str(e))

    @staticmethod
    def calc_metric(true_ref, test_ref, metric):
        """ Wrapper from skimage psnr and ssim comparison. """
        if isinstance(true_ref, (Path, str)):
            true_ref = Image.open(true_ref)
        if isinstance(test_ref, (Path, str)):
            test_ref = Image.open(test_ref)

        true_ref = np.array(true_ref)
        test_ref = np.array(test_ref)

        if metric == Metrics.PSNR:
            result = compare_psnr(true_ref, test_ref)
        elif metric == Metrics.SSIM:
            result = compare_ssim(true_ref, test_ref, multichannel=True)
        elif metric == Metrics.MSSSIM:
            # TODO: find official implementation that really works (skvideo
            # did not work and apparently it's not being supported anymore)
            # TODO: verify this implementation is correct. Compare with the
            # other implementations
            true_ref = np.expand_dims(
                np.expand_dims(true_ref, 0), 0).swapaxes(0, -1)
            test_ref = np.expand_dims(
                np.expand_dims(test_ref, 0), 0).swapaxes(0, -1)
            aux_result = list(map(compare_msssim, true_ref, test_ref))
            result = np.mean(aux_result)

        # Put a huge number in this case
        result = np.iinfo(np.uint8).max if result == float('inf') else result

        return result

    @staticmethod
    def save_img(img_ref, path, mode='RGB'):
        """ Method that receives a numpy array and a path. It saves an image
            through PIL
        """
        try:
            if isinstance(img_ref, (str, PosixPath)):
                img_ref = Image.open(img_ref)
            elif isinstance(img_ref, np.ndarray):
                img_ref = Image.fromarray(img_ref)

            img_ref.save(path, mode=mode)
            img_ref.close()
        except Exception as e:
            print(str(e))

    @staticmethod
    def get_size(path):
        """ Wrapper that receives a path of an image and returns its size """
        with Image.open(path) as img_ref:
            width, height = img_ref.size
        return width, height

    @staticmethod
    def save_jpeg_analysis(img_path, out_pathname):
        """ This method analysis an image with jpeg wrt to bpp and visual
            metrics. It saves a csv file with this info.
        """
        # PIL doesn't handle numpy types
        quality = list(map(int, np.arange(69, 73)))
        params = list(map(lambda q: dict(quality=q), quality))

        valid, img_ref = ImgProc.is_pillow_valid_img(img_path)
        if not valid:
            return

        pixel_num = np.prod(img_ref.size)
        img_data = np.array(img_ref)
        bpp, psnr, ssim, msssim = [], [], [], []
        for param in params:
            with BytesIO() as buffer:
                img_ref.save(buffer, 'jpeg', **param)
                codec_data = np.array(Image.open(buffer))
                codec_buffer = buffer.getvalue()
            num_bits = Bits(codec_buffer).length
            bpp.append(num_bits / pixel_num)
            psnr.append(ImgProc.calc_metric(img_data, codec_data, Metrics.PSNR))
            ssim.append(ImgProc.calc_metric(img_data, codec_data, Metrics.SSIM))
            msssim.append(ImgProc.calc_metric(img_data, codec_data,
                                              Metrics.MSSSIM))
        data = np.stack((quality, psnr, ssim, msssim, bpp), axis=1)
        df = pd.DataFrame(data, columns=['quality', 'psnr', 'ssim',
                                         'msssim', 'bpp'])
        df.set_index('quality', inplace=True)
        df.to_csv(out_pathname)

    @staticmethod
    def save_jpeg2000_kakadu(orig_pathname, output_pathname, bpp):
        """ Save a image in kakadu jpeg2000 format """
        cmd = 'kdu_compress -i \'{}\' -o \'{}\' -rate {} >/dev/null 2>&1'\
            .format(orig_pathname, output_pathname, bpp)
        if os.system(cmd):
            raise RuntimeError('Error when executing kakadu binary!')

my_analysis=ImgProc()


save_jpeg_analysis='/home/marcelo/Documentos/rnn-conv-autoencoder/imagens_teste/kodim'

for j in range(13,14):

	original='/home/marcelo/Documentos/rnn-conv-autoencoder/imagens_teste/kodim'
	if j<10:
		original_instance=original+'0'+str(j)+'.bmp'
	else:
		original_instance=original+str(j)+'.bmp'
	my_analysis.save_jpeg_analysis(original_instance,'analise'+str(j)+'.csv')

#print(original)
#comp='/home/marcelo/Documentos/rnn-conv-autoencoder/Resultados/resultados_BMP_7epochs/kodim'
#print(comp)
#msssim=[]
#for j in range (0,16):
#    msssim=[]
#    for i in range(1,25):
#        if i<10:  
#            original_instance= original+'0'+str(i)+'.bmp'
#        else:
#            original_instance= original+str(i)+'.bmp'
#        if j<10:
#            comp_instance = comp+str(i)+'0'+str(j)+'.bmp'
#        else:
#            comp_instance = comp+str(i)+str(j)+'.bmp'
     
#        r=my_analysis.calc_metric(original_instance,comp_instance,1)
#        msssim.append(r)
 #   print((str(np.mean(msssim))).replace('.',','))
