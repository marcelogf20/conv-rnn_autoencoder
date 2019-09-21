import pytorch_ssim
import torch
from torch.autograd import Variable
from torch import optim
import cv2
import numpy as np
from PIL import Image

npImg1 = cv2.imread("einstein.png")
img   = Image.open('einstein.png')
ycbcr = img.convert('YCbCr')


img1 = torch.from_numpy(np.array(npImg1)).float()/255.0
img2 = torch.from_numpy(np.array(ycbcr)).float()/255.0

img1 = img1[None, :, :]
img2 = img2[None, :, :]

print(img1.size())
img3 = torch.rand(img1.size())

if torch.cuda.is_available():
    img1 = img1.cuda()
    img2 = img2.cuda()


img1 = Variable(img1,  requires_grad=False)
img2 = Variable(img2, requires_grad = True)
img3 = Variable(img3, requires_grad = True)


# Functional: pytorch_ssim.ssim(img1, img2, window_size = 11, size_average = True)
ssim_value = pytorch_ssim.ssim(img1, img3)
ssim_value2 = pytorch_ssim.ssim(img1, img2)
print("Initial ssim RGB:", ssim_value)
print("Initial ssim YUV:", ssim_value2)
'''
# Module: pytorch_ssim.SSIM(window_size = 11, size_average = True)
ssim_loss = pytorch_ssim.SSIM()

optimizer = optim.Adam([img2], lr=0.01)

while ssim_value < 0.95:
    optimizer.zero_grad()
    ssim_out = -ssim_loss(img1, img2)
    ssim_value = - ssim_out.data[0]
    print(ssim_value)
    ssim_out.backward()
    optimizer.step()
'''