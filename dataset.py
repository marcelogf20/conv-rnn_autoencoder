# modified from https://github.com/desimone/vision/blob/fb74c76d09bcc2594159613d5bdadd7d4697bb11/torchvision/datasets/folder.py

import os
import os.path

import torch
from torchvision import transforms
import torch.utils.data as data
from PIL import Image
import glob

IMG_EXTENSIONS = [
    '.jpg',
    '.JPG',
    '.jpeg',
    '.JPEG',
    '.png',
    '.PNG',
    '.ppm',
    '.PPM',
    '.bmp',
    '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def default_loader(path):
    return Image.open(path).convert('RGB')

class ConcatDataset(torch.utils.data.Dataset):

    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)
    


class ImageFolderYCbCr(data.Dataset):
    """ ImageFolder can be used to load images where there are no labels."""

    def __init__(self, root, transform=None):
        images = []
        for filename in os.listdir(root):
            for files in glob.glob('%s/*.*' % (root+'/'+filename)): 
                images.append(files)

        self.root = root
        self.imgs = images
        self.transform = transform

    def __getitem__(self, index):
        path  = self.imgs[index]
        img   = Image.open(path)
        ycbcr = img.convert('YCbCr')
        #ycbcr = np.array(ycbcr)
        if self.transform is not None:
            ycbcr = self.transform(ycbcr)
    
        return ycbcr

    def __len__(self):
        return len(self.imgs)



class ImageFolderRGB(data.Dataset):
    """ ImageFolder can be used to load images where there are no labels."""

    def __init__(self, root, transform=None):
        images = []
        for filename in os.listdir(root):
            for files in glob.glob('%s/*.*' % (root+'/'+filename)): 
                images.append(files)

        self.root = root
        self.imgs = images
        self.transform = transform

    def __getitem__(self, index):
        path  = self.imgs[index]
        img   = Image.open(path)
        img = img.convert('RGB')

        #ycbcr = np.array(ycbcr)
        if self.transform is not None:
            img = self.transform(img)
    
        return img

    def __len__(self):
        return len(self.imgs)

class BSDS500Crop128(data.Dataset):
    def __init__(self, folder_path,train_transform):
        self.files = sorted(glob.glob('%s/*.*' % folder_path))
        self.train_transform = train_transform

    def __getitem__(self, index):
        path = self.files[index % len(self.files)]
        img = Image.open(path)
        ycbcr = img.convert('YCbCr')
        #ycbcr = np.array(ycbcr)
        ycbcr = self.train_transform(ycbcr)

        return ycbcr

    def __len__(self):
        return len(self.files)
 


