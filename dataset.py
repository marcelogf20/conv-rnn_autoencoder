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
    


class ImageFolder(data.Dataset):
    """ ImageFolder can be used to load images where there are no labels."""

    def __init__(self, root, transform=None, loader=default_loader):
        images = []
        for filename in os.listdir(root):
            if is_image_file(filename):
                images.append('{}'.format(filename))

        self.root = root
        self.imgs = images
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        filename = self.imgs[index]
        try:
            img = self.loader(os.path.join(self.root, filename))
        except:
            return torch.zeros((3, 32, 32))

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
        rgb = np.array(img, dtype=np.uint8)
        yuv =  self.train_transform(yuv)

        return yuv

    def __len__(self):
        return len(self.file)