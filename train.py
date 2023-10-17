import argparse
import logging
import os
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split,ConcatDataset
from tqdm import tqdm
from sklearn.model_selection import KFold
import numpy as np
import wandb
from evaluate import evaluate
from unet import UNet
from utils.data_loading import BasicDataset, OCT
from utils.dice_score import dice_loss
from utils.utils import plot_img_and_mask, save_image_overlay


def train_model(
        model,
        config,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
):
    # 1. Create dataset
    train_dataset =  OCT(config.dataset_dir,True, config.scale)
    # 2. Creating dirs to save outputs
    train_output_path = os.path.join(config.load_dir,'train_results')
    if not os.path.exists(train_output_path):
        os.mkdir(train_output_path)
    eval_output_path = os.path.join(config.load_dir,'evaluation_results')
    if not os.path.exists(eval_output_path):
        os.mkdir(eval_output_path)
    
    # (Initialize logging)
    experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')
    experiment.config.update(
        dict(epochs=config.epochs, batch_size=config.batch_size, learning_rate=config.lr,
             val_percent=config.val, save_checkpoint=config.load_dir, img_scale=config.scale, amp=config.amp)
    )
    
    logging.info(f'''Starting training:
        Epochs:          {config.epochs}
        Batch size:      {config.batch_size}
        Learning rate:   {config.lr}
        Training size:   {15}
        Validation size: {5}
        Checkpoints:     {config.load_dir}
        Device:          {device.type}
        Images scaling:  {config.scale}
        Mixed Precision: {config.amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(model.parameters(),
                lr=config.lr, weight_decay=weight_decay, momentum=momentum)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=config.amp)
    criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()
    global_step = 0

    # 5. Begin training
    # 4-fold Cross Validation model evaluation
    kfold = KFold(n_splits=4, shuffle=True)
    fold_results = {}
    print("testing ssh")
    for fold, (train_ids, val_ids) in enumerate(kfold.split(np.arange(len(train_dataset)))):
        
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)
        
        # Define data loaders for training and testing data in this fold
        train_loader = torch.utils.data.DataLoader(
                        train_dataset, 
                        batch_size=args.batch_size, sampler=train_subsampler)
        val_loader = torch.utils.data.DataLoader(
                        train_dataset,
                        batch_size=args.batch_size, sampler=val_subsampler)
        
        for epoch in range(1, config.epochs + 1):
            model.train()
            epoch_loss = 0
            for iter,batch in enumerate(train_loader):
                images, true_masks = batch['image'], batch['mask']

                assert images.shape[1] == model.n_channels, \
                    f'Network has been defined with {model.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long)
                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=config.amp):
                    masks_pred = model(images)
                    if model.n_classes == 1:
                        ce_loss = criterion(masks_pred.squeeze(1), true_masks.float())
                        dice = dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
                    else:
                        ce_loss = criterion(masks_pred, true_masks)
                        dice = dice_loss(
                            F.softmax(masks_pred, dim=1).float(),
                            F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                            multiclass=True
                        )
                loss = 20*ce_loss + dice*200
                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()
                global_step += 1
                epoch_loss += loss.item()
                if iter % config.save_frq ==0:
                    masks_pred = torch.argmax(masks_pred,1)
                    img = images.detach().squeeze().cpu().numpy()
                    mask = masks_pred.detach().squeeze().cpu().numpy()
                    save_image_overlay(img,mask,train_output_path,iter)

                experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })
                logging.info(f"Fold:{fold} Epoch:{epoch} Loss:{loss.item():.3f} CE Loss:{ce_loss.item():.3f} Dice Loss:{dice.item():.3f}")
                
            logging.info(f"After {epoch} Epochs Loss:{epoch_loss:.3f}")
            # Evaluation round
            histograms = {}
            for tag, value in model.named_parameters():
                tag = tag.replace('/', '.')
                if not (torch.isinf(value) | torch.isnan(value)).any():
                    histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                if not (torch.isinf(value.grad) | torch.isnan(value.grad)).any():
                    histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

            val_score = evaluate(model, val_loader, device, config.amp)
            scheduler.step(val_score)
            logging.info(f'Validation Dice score: {val_score:.3f}')
           
            try:
                experiment.log({
                    'learning rate': optimizer.param_groups[0]['lr'],
                    'validation Dice': val_score,
                    'images': wandb.Image(images[0].cpu()),
                    'masks': {
                        'true': wandb.Image(true_masks[0].float().cpu()),
                        'pred': wandb.Image(masks_pred.argmax(dim=1)[0].float().cpu()),
                    },
                    'step': global_step,
                    'epoch': epoch,
                    **histograms
                })
            except:
                pass
        fold_results[fold] = val_score.item()
        if config.load_dir:
            state_dict = model.state_dict()
            torch.save(state_dict, os.path.join(args.load_dir, f'checkpoint_fold_{fold}.pth'))
            logging.info(f'Checkpoint {fold} saved!')
    
    logging.info(f'Training Complete!!!')
    logging.info(f'Validation Scores for each fold{fold_results}')

        


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--dataset-dir', type=str, default=r'C:\MyProjects\macuject\data', 
                        help='dataset directory')
    parser.add_argument('--load-dir', '-f', type=str, default=r'C:\MyProjects\macuject\large_weight',
                         help='Model saving path')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=5.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    parser.add_argument('--device', '-d', type=str, default='cpu', help='cpu/gpu')
    parser.add_argument('--save-frq', '-sf', type=int, default=5, help='Image save frequency')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device(args.device)
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    model = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    model = model.to(memory_format=torch.channels_last)

    logging.info(f'Network:\n'
                 f'\t{model.n_channels} input channels\n'
                 f'\t{model.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')

    if os.path.exists(args.load_dir):
        try:
            state_dict = torch.load(args.load_dir, map_location=device)
            del state_dict['mask_values']
            model.load_state_dict(state_dict)
            logging.info(f'Model loaded from {args.load_dir}')
        except:
            logging.info(f'No model checkpoint found in {args.load_dir}')
    else:
        logging.info(f'Model not found. Creating the output directory to save model {args.load_dir}')
        os.mkdir(args.load_dir)

    model.to(device=device)
    try:
        train_model(
            model=model,
            config=args
        )
    except torch.cuda.OutOfMemoryError:
        logging.error('Detected OutOfMemoryError! '
                      'Enabling checkpointing to reduce memory usage, but this slows down training. '
                      'Consider enabling AMP (--amp) for fast and memory efficient training')
        torch.cuda.empty_cache()
        model.use_checkpointing()
        train_model(
            model=model,
            config=args
        )
