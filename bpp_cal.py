import os, os.path
from PIL import Image
import glob
import numpy as np

def calc_size(path):	
	size_list=[]
	i=0
	filenames=glob.glob(path)
	for filename in filenames: #assuming gif
	    size_list.append(os.path.getsize(filename))
	    i+=1;
	return size_list

X=1
for j in range(16,17):
	path='imgs_codificadas/imgs_codificadas_bmp/*iter'+str(j)+'.npz'
	path2='imagens_teste/*.bmp'
	
	size_cod= calc_size(path)
	size_cod_mean= np.mean(size_cod)
	size_img= np.array(calc_size(path2))
	
	bpp_mean = size_cod_mean/147456

	print('média',str(bpp_mean).replace(".", ","))

	r=np.divide(size_cod,147456)


	for size in size_cod:
		bpp = size*8/(1179648)
		print((str(bpp)).replace(".", ","))

	print('media:',str(np.mean(r)).replace(".", ",")) 