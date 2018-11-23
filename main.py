import sys

#sys.stdout = open("log.txt", 'a')

from model import UNet3D
from data import MRIDataset
import argparse
import math
import random
import shutil
import numpy as np
import os

import torch

from torch.utils.data import DataLoader 
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import BCELoss
from torch.nn.parallel import DataParallel
from torch.nn.utils import clip_grad_norm_

from cumulative_average import CumulativeAverager

# Dataset Folder Locations

folder_params = {'tcia_folder':'data/NiFTiSegmentationsEdited', 'brats_folder':'data/BraTs', 
                    'hgg_folder':'data/Pre-operative_TCGA_GBM_NIfTI_and_Segmentations',
                    'lgg_folder':'data/Pre-operative_TCGA_LGG_NIfTI_and_Segmentations'}

# Command Line Arguments

ap = argparse.ArgumentParser()
ap.add_argument('--batch_size', type=int, help='Number of 3D voxel batches')
ap.add_argument('--lr', type=float, help='Initial Learning rate', default=0.001)
ap.add_argument('--lr_decay', type=float, help='Learning Rate Decay', default=0.1)
ap.add_argument('--optimizer', help='sgd, adam', default='adam')
ap.add_argument('--epochs', help='Total number of epochs to train on data', default=200, type=int)
ap.add_argument('--iters', help='Number of training batches per epoch', default=None, type=int)
ap.add_argument('--aug', action='store_true', help='Flag to decide about input augmentations')
ap.add_argument('--demo', action='store_true', help='Flag to indicate testing')
ap.add_argument('--load_from', help='Path to checkpoint dict', default=None)
args = ap.parse_args()

data_strs = ['lgg_t1', 'lgg_t2', 'lgg_flair', 'hgg_t1', 'hgg_t2', 'hgg_flair']

log_str = ''
if os.path.exists('log.txt'):
    with open('log.txt', 'r') as f:
        log_str = f.read()

def add_to_log(st):
    global log_str
    print (st)
    log_str = log_str+'\n'+st
    with open('log.txt', 'w') as f:
        f.write(log_str)
    return log_str

# Choices of datasets: can be any subset of the following types of MRI
# [LGG: t1->0, t2->1, flair->2; HGG: t1->3, t2->4, flair->5]

choice, losstype = [0,1,2,3,4,5], 'bce'

def get_model(mode, flag_3d = True, channel_size_3d = 32, mri_slice_dim = 256, choice=[0,1,2,3,4,5]):
    
    assert math.log(mri_slice_dim, 2).is_integer() # Image dims must be powers of 2 

    if mode == 'train' and args.aug:
        aug = True
    else:
        aug = False

    t1_lgg = MRIDataset(**folder_params, type_str='T1', stage='lgg', mode=mode, 
                channel_size_3d=channel_size_3d, flag_3d=flag_3d, mri_slice_dim=mri_slice_dim, aug=aug)
    t2_lgg = MRIDataset(**folder_params, type_str='T2', stage='lgg', mode=mode, 
                channel_size_3d=channel_size_3d, flag_3d=flag_3d, mri_slice_dim=mri_slice_dim, aug=aug)
    flair_lgg = MRIDataset(**folder_params, type_str='FLAIR', stage='lgg', mode=mode, 
                channel_size_3d=channel_size_3d, flag_3d=flag_3d, mri_slice_dim=mri_slice_dim, aug=aug)

    t1_hgg = MRIDataset(**folder_params, type_str='T1', stage='hgg', mode=mode, 
                channel_size_3d=channel_size_3d, flag_3d=flag_3d, mri_slice_dim=mri_slice_dim, aug=aug)
    t2_hgg = MRIDataset(**folder_params, type_str='T2', stage='hgg', mode=mode, 
                channel_size_3d=channel_size_3d, flag_3d=flag_3d, mri_slice_dim=mri_slice_dim, aug=aug)
    flair_hgg = MRIDataset(**folder_params, type_str='FLAIR', stage='hgg', mode=mode, 
                channel_size_3d=channel_size_3d, flag_3d=flag_3d, mri_slice_dim=mri_slice_dim, aug=aug)
    
    if 0 in choice:
        dataset = t1_lgg
    if 1 in choice:
        dataset.segmentation_pairs.extend(t2_lgg.segmentation_pairs)
    if 2 in choice:
        dataset.segmentation_pairs.extend(flair_lgg.segmentation_pairs)
    if 3 in choice:
        dataset.segmentation_pairs.extend(t1_hgg.segmentation_pairs)
    if 4 in choice:
        dataset.segmentation_pairs.extend(t2_hgg.segmentation_pairs)
    if 5 in choice:
        dataset.segmentation_pairs.extend(flair_hgg.segmentation_pairs)
    
    return dataset 


if not args.demo:
    primary_dataset = get_model('train')
    val_dataset =  get_model('val')
    
    primary_data_loader = DataLoader(primary_dataset, batch_size=1, shuffle=True)
    val_data_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)
else:
    primary_dataset = get_model('test')
    primary_data_loader = DataLoader(primary_dataset, shuffle=False, batch_size=1)

# Book-keeping

types = ', '.join([data_strs[i] for i in choice])
prep = lambda x: 'Number of %s in MRI study %s: %d\n'%('patients' if x.flag_3d else 'slices', types, len(x))
log_str = add_to_log(prep(primary_dataset))
log_str = add_to_log(prep(val_dataset))
# Define 3D UNet and train, val, test scripts

net = UNet3D()
net.train()

net = DataParallel(net.cuda())
bce_criterion = BCELoss()

def get_optimizer(st, lr, momentum=0.9):
    if st == 'sgd':
        return SGD(net.parameters(), lr = lr, momentum=momentum)
    elif st == 'adam':
        return Adam(net.parameters(), lr = lr)

optimizer = get_optimizer(args.optimizer, args.lr)
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=2)

def dice_loss(y, pred):
    smooth = 1.

    yflat = y.view(-1)
    predflat = pred.view(-1)
    intersection = (yflat * predflat).sum()
    
    return 1 - ((2. * intersection + smooth) /
              (yflat.sum() + predflat.sum() + smooth))

def train(epoch, losstype='dice'):
    
    assert len(choice)>0

    if args.iters is not None:
        primary_data_loader.dataset.segmentation_pairs = primary_data_loader.dataset.segmentation_pairs[:args.iters]
    
    for idx, (inp, seg) in enumerate(primary_data_loader):
        
        optimizer.zero_grad()
        
        inp, seg = torch.tensor(inp).cuda(), torch.tensor(seg, requires_grad=False).cuda()
        
        out = net.forward(inp)
        
        ''' 
        check = out.cpu().detach()
        check.apply_(lambda x: bool(x>=0. and x<=1.))
        check = torch.tensor(check, dtype=torch.uint8) 
        print (check.all())
        
        print (torch.min(out), torch.max(out))
        '''

        if losstype=='dice':
            loss = dice_loss(seg, out)
        else:
            loss = bce_criterion(out, seg)
        
        avg_tool.update(loss)
        log_str = add_to_log("Epoch %d, Batch %d/%d: Loss=%0.6f"%(epoch, idx+1, len(primary_data_loader), avg_tool.get_average()))
        loss.backward()
        optimizer.step()

        clip_grad_norm_(net.parameters(), 5.0)

def validate(losstype): #num_patients=20):
    
    net.eval()
    val_loss_avg = CumulativeAverager()
    
    add_to_log("Performing validation test on %d samples"%len(val_data_loader))
    
    if args.iters is not None:
        val_data_loader.dataset.segmentation_pairs = val_data_loader.dataset.segmentation_pairs[:args.iters]

    with torch.no_grad():
        
        for inp, seg in val_data_loader:
            
            inp, seg = torch.tensor(inp).cuda(), torch.tensor(seg).cuda()
            out = net.forward(inp)
            if losstype == 'bce':
                loss = bce_criterion(out, seg)
            else:
                loss = dice_loss(seg, out)
            val_loss_avg.update(loss)
    
    val_loss = val_loss_avg.get_average()
    log_str = add_to_log('Validation Loss=%0.6f'%(val_loss))

    return val_loss

def test():
    pass

def saver_fn(net_params, is_best, name='checkpt.pth.tar'):
    torch.save(net_params, name)
		
    if is_best is not None:
        shutil.copyfile(name, 'checkpt_best_%d.pth.tar'%(is_best))

# Train or Test 


if not args.demo:
    
    avg_tool = CumulativeAverager()

    vloss, is_best = torch.tensor(float(np.inf)), None
    if args.load_from is not None:
        if os.path.isfile(args.load_from):
            log_str = add_to_log("=> loading checkpoint '{}'".format(args.load_from))
            checkpoint = torch.load(args.load_from)
            start = checkpoint['epoch']
            vloss = checkpoint['best_val_loss']
            net.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.lr = checkpoint['learning_rate']
            log_str = add_to_log("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.load_from, checkpoint['epoch']))
        else:
            log_str = add_to_log("=> no checkpoint found at '{}'".format(args.load_from)) 
    else:
        start = 0
    
    for epoch in range(start, args.epochs):

        train(epoch, losstype=losstype)
        val_loss = validate(losstype=losstype).cpu()
        scheduler.step(val_loss)
		
        if vloss > val_loss:
            vloss = val_loss
            is_best = epoch+1
       
        for param_group in optimizer.param_groups:
            lr = param_group['lr']

        saver_fn({
            'epoch': epoch + 1,
            'state_dict': net.state_dict(),
            'best_val_loss': vloss,
            'optimizer' : optimizer.state_dict(),
			'learning_rate': lr
        }, is_best) 
        
else:
    test()


