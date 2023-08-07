import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from utils.data_loading import BasicDataset
from unet import UNet
from utils.utils import plot_img_and_mask, save_image_overlay
import os
import glob

def input_files(file_dir):
    files = glob.glob(os.path.join(file_dir,'*.tif'))
    return files


def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(full_img, scale_factor, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img).cpu()
        output = F.interpolate(output, (full_img.size[1], full_img.size[0]), mode='bilinear')
        if net.n_classes > 1:
            mask = output.argmax(dim=1)
        else:
            mask = torch.sigmoid(output) > out_threshold

    return mask[0].long().squeeze().numpy()


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default=r'C:\MyProjects\macuject\large_weight\checkpoint_fold_2.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', default=r'C:\MyProjects\macuject\data\test\images',
                         help='Input Images path')
    parser.add_argument('--output', '-o', default=r'C:\MyProjects\macuject\large_weight\predictions', help='Output saving path')
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=0.5,
                        help='Scale factor for the input images')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    return parser.parse_args()


def get_output_filenames(input_files,output_path):
    names = [f'{os.path.basename(name).split(".")[0]}_OUT.tif' for name in input_files]
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    names = [os.path.join(output_path,filename) for filename in names] 
    return names


def mask_overlay(mask: np.ndarray,image):
    if mask.ndim == 3:
        mask = np.argmax(mask, axis=0)
    mask = (mask/mask.max())*255
    mask = mask.astype(np.uint8)
    overlay = save_image_overlay(image,mask)
    return overlay


if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    in_files = input_files(args.input)
    out_files = get_output_filenames(in_files,args.output)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')
    model = UNet(n_channels=3, n_classes=args.classes, bilinear=False)
    model = model.to(memory_format=torch.channels_last)
    state_dict = torch.load(args.model, map_location=device)
    #mask_values = state_dict.pop('mask_values', [0, 1])
    model.load_state_dict(state_dict)

    logging.info('Model loaded!')

    for i, filename in enumerate(in_files):
        logging.info(f'Predicting image {filename} ...')
        img = Image.open(filename)

        mask = predict_img(net=model,
                           full_img=img,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           device=device)

        out_filename = out_files[i]
        result = mask_overlay(mask,np.asarray(img))
        result.save(out_filename)
        logging.info(f'Mask saved to {out_filename}')

        if args.viz:
            logging.info(f'Visualizing results for image {filename}, close to continue...')
            plot_img_and_mask(img, mask)
