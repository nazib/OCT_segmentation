import logging
import numpy as np
import torch
from PIL import Image
from functools import lru_cache
from functools import partial
from itertools import repeat
from multiprocessing import Pool
import os
from os import listdir
from os.path import splitext, isfile, join
import glob
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm
from PIL import Image

def load_image(filename):
    ext = splitext(filename)[1]
    if ext == '.npy':
        return Image.fromarray(np.load(filename))
    elif ext in ['.pt', '.pth']:
        return Image.fromarray(torch.load(filename).numpy())
    else:
        return Image.open(filename)

class BasicDataset(Dataset):
    def __init__(self, dataset_dir: str,istrain:bool, scale: float = 1.0,
                 gt_suffix:str='', mask_suffix: str = ''):
        self.root = dataset_dir
        self.isTrain = istrain
        if self.isTrain:
            self.images_dir = os.path.join(self.root,'training','images') 
            self.mask_dir = os.path.join(self.root,'training','mask')
            self.gt_dir = os.path.join(self.root,'training','1st_manual')
        else:
            self.images_dir = os.path.join(self.root,'test','images') 
            self.mask_dir = os.path.join(self.root,'test','mask')
        
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix
        self.gt_suffix = gt_suffix
        self.ids = [file.split(os.sep)[-1].split('_')[0] for file in listdir(self.images_dir) 
                    if isfile(join(self.images_dir, file)) and not file.startswith('.')]
        
        if not self.ids:
            raise RuntimeError(f'No input file found in {self.images_dir}, make sure you put your images there')

        logging.info(f'Creating dataset with {len(self.ids)} examples')
        
    def __len__(self):
        return len(self.ids)
    
    @staticmethod
    def masking(img,mask):
        img = img*mask
        return img

    @staticmethod
    def preprocess(pil_img, scale, is_mask):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img = np.asarray(pil_img)
        
        if is_mask:
            mask = np.zeros((newH, newW), dtype=np.int64)
            for i, v in enumerate(np.unique(img)):
                if img.ndim == 2:
                    mask[img == v] = i
                else:
                    mask[(img == v).all(-1)] = i
            return mask

        else:
            if img.ndim == 2:
                img = img[np.newaxis, ...]
            else:
                img = img.transpose((2, 0, 1))

            if (img > 1).any():
                img = img / img.max()
            return img

    def __getitem__(self, idx):
        id = self.ids[idx]
        img_file = glob.glob(os.path.join(self.images_dir,id)+"*.*")
        mask_file = glob.glob(os.path.join(self.mask_dir,id)+"*.*")
        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {id}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {id}: {mask_file}'
        if self.isTrain:
            gt_file = glob.glob(os.path.join(self.gt_dir,id)+"*.*")
            assert len(img_file) == 1, f'Either no GT or multiple GT images found for the ID {id}: {gt_file}'
            gt = load_image(gt_file[0])

        mask = load_image(mask_file[0])
        img = load_image(img_file[0])
        
        assert img.size == mask.size, \
            f'Image and mask {id} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img, self.scale, is_mask=False)
        mask = self.preprocess( mask, self.scale, is_mask=True)
        img = self.masking(img,mask)

        if self.isTrain:
            gt = self.preprocess( gt, self.scale, is_mask=True)
            return {
                'image': torch.as_tensor(img.copy()).float().contiguous(),
                'mask': torch.as_tensor(gt.copy()).long().contiguous()
            }
        else:
             return {
                'image': torch.as_tensor(img.copy()).float().contiguous()
            }

class OCT(BasicDataset):
    def __init__(self, dataset_dir,istrain, scale=1):
        super().__init__(dataset_dir,istrain, scale, mask_suffix='_mask')
