import pandas as pd
import numpy as np
import glob
    
iteracoes=16
num_imgs =24
psnr=np.zeros((num_imgs,iteracoes))
bpp_array=np.zeros((num_imgs,iteracoes))
quality=np.zeros((num_imgs,iteracoes))
path='./'

ssim=np.zeros((num_imgs,iteracoes))
ms_ssim=np.zeros((num_imgs,iteracoes))
entrada=  np.arange(0.125,2.1,0.125)
i=0
for filename in glob.glob(path+'*.csv'):
    df = pd.read_csv(filename)
    j=0
    for bpp in entrada:
        df2=df.loc[(df['bpp']-bpp).abs().argsort()[:1]]
        bpp_array[i][j] = float(df2['bpp'])
        ssim[i][j] =  np.array(df2['ssim'])
        quality[i][j] =  np.array(df2['quality'])
        ms_ssim[i][j]= np.array(df2['msssim'])
        psnr[i][j]= np.array(df2['psnr'])
        j+=1
    i+=1

print('quality:')
for y in range(iteracoes):
    ls=[]
    for x in range(num_imgs):
        ls.append(quality[x][y])
    print(str(np.mean(ls)).replace('.',','))
    
    
print('bpp:')
for y in range(iteracoes):
    ls=[]
    for x in range(num_imgs):
        ls.append(bpp_array[x][y])
    print(str(np.mean(ls)).replace('.',','))
    
    
print('PSNR:')
for y in range(iteracoes):
    ls=[]
    for x in range(num_imgs):
        ls.append(psnr[x][y])
    print(str(np.mean(ls)).replace('.',','))

        
print('SSIM:')
for y in range(iteracoes):
    ls=[]
    for x in range(num_imgs):
        ls.append(ssim[x][y])
    print(str(np.mean(ls)).replace('.',','))
    
print('MSSSIM:')
for y in range(iteracoes):
    ls=[]
    for x in range(num_imgs):
        ls.append(ms_ssim[x][y])
    print(str(np.mean(ls)).replace('.',','))
