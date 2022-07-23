from os import listdir
from os.path import join
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image, ImageOps
import random
import torch


def random_patch(img1, scale_x, scale_y, tp):
    # (ih, iw) = img1.size
    # tp = imageSize
    ix = scale_x
    iy = scale_y
    # ix = random.randrange(0, iw - tp)
    # iy = random.randrange(0, ih - tp)
    img1 = img1.crop((ix, iy, ix + tp, iy + tp))
    # img2 = img2.crop((iy, ix, iy + tp, ix + tp))

    info_patch = {
        'x_start': ix, 'y_start': iy, 'x_end': ix + tp, 'y_end': iy + tp}

    return img1, info_patch


def augment(img_in, flip_factor, mirror_factor, rotate_factor, flip_h=True, rot=True):
    info_aug = {'flip_h': False, 'flip_v': False, 'trans': False}

    if flip_factor < 0.5 and flip_h:
        img_in = ImageOps.flip(img_in)
        info_aug['flip_h'] = True

    if rot:
        if mirror_factor < 0.5:
            img_in = ImageOps.mirror(img_in)
            info_aug['flip_v'] = True
        if rotate_factor < 0.5:
            img_in = img_in.rotate(180)
            info_aug['trans'] = True

    return img_in, info_aug


class LoadDataset(Dataset):
    def __init__(self, root, h_im_dir, l_im_dir, image_size, height, width, data_augmentation=True,
                 transform=None):
        super(LoadDataset).__init__()
        self.org_im_name = [join(root, h_im_dir, x) for x in listdir(join(root, h_im_dir))]
        self.target_im_name = [join(root, l_im_dir, x) for x in listdir(join(root, l_im_dir))]
        self.org_im_name.sort()
        self.target_im_name.sort()
        self.transform = transform
        self.imag_size = image_size
        self.data_augmentation = data_augmentation
        self.scale_x = 0
        self.scale_y = 0
        self.height = height
        self.width = width

    def __getitem__(self, index):
        tp = self.imag_size
        iw = self.width
        ih = self.height
        self.scale_x = random.randrange(0, iw - tp)
        self.scale_y = random.randrange(0, ih - tp)
        self.flip_factor = random.random()
        self.mirror_factor = random.random()
        self.rotate_factor = random.random()
        org_img = Image.open(self.org_im_name[index])
        target_img = Image.open(self.target_im_name[index])
        org_img, _ = random_patch(org_img, scale_x=self.scale_x, scale_y=self.scale_y,
                                  tp=self.imag_size)
        target_img, _ = random_patch(target_img, scale_x=self.scale_x, scale_y=self.scale_y,
                                     tp=self.imag_size)
        org_img, _ = augment(org_img, flip_factor=self.flip_factor, mirror_factor=self.mirror_factor,
                             rotate_factor=self.rotate_factor)
        target_img, _ = augment(target_img, flip_factor=self.flip_factor, mirror_factor=self.mirror_factor,
                                rotate_factor=self.rotate_factor)

        return org_img, target_img

    def __len__(self):
        return len(self.org_im_name)
