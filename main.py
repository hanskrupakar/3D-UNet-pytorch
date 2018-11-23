import sys
sys.stdout = open("log.txt", 'w')

from model import UNet3D
from data import MRIDataset
import argparse
import math
import random
import shutil
import numpy as np

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
ap.add_argument('--aug', action='store_true', help='Flag to decide about input augmentations')
ap.add_argument('--demo', action='store_true', help='Flag to indicate testing')
args = ap.parse_args()

def get_models(mode, flag_3d = True, channel_size_3d = 32, mri_slice_dim = 256):
    
    assert math.log(mri_slice_dim, 2).is_integer() # Image dims must be powers of 2 

    if mode == 'train' and args.aug:
        aug = True
    else:
        aug = False

    # Segregated dataset handlers for easier plug and play

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
    
    dataset_handlers = [t1_lgg, t2_lgg, flair_lgg, t1_hgg, t2_hgg, flair_hgg]
    return dataset_handlers 

get_dataset_loaders = lambda x, y: [DataLoader(item, batch_size=1, shuffle=y) for item in x]

if not args.demo:
    primary_dataset_handlers = get_models('train')
    val_dataset_handlers =  get_models('val')
    
    primary_dataset_loaders = get_dataset_loaders(primary_dataset_handlers, True)
    val_dataset_loaders = get_dataset_loaders(val_dataset_handlers, False)
else:
    primary_dataset_handlers = get_models('test')
    primary_dataset_loaders = get_dataset_loaders(test_dataset_handlers, False)

# Book-keeping

prep = lambda x: 'Number of %s in %s study for %s: %d\n'%('patients' if x.flag_3d else 'slices', x.type, x.stage, len(x))
for dl in primary_dataset_handlers:
    print (prep(dl), flush=True)

# Define 3D UNet and train, val, test scripts

net = UNet3D()
net.train()
cpu_net = UNet3D()
cpu_net.eval()

net = DataParallel(net.cuda())
bce_criterion = BCELoss()

def get_optimizer(st, lr, momentum=0.9):
    if st == 'sgd':
        return SGD(net.parameters(), lr = lr, momentum=momentum)
    elif st == 'adam':
        return Adam(net.parameters(), lr = lr)

optimizer = get_optimizer(args.optimizer, args.lr)
scheduler = ReduceLROnPlateau(optimizer, 'min')

def dice_loss(y, pred):
    smooth = 1.

    yflat = y.view(-1)
    predflat = pred.view(-1)
    intersection = (yflat * predflat).sum()
    
    return 1 - ((2. * intersection + smooth) /
              (yflat.sum() + predflat.sum() + smooth))

def train(epoch, epoch_size=200, choice=[0,1,2,3,4,5], losstype='bce'):
    
    assert len(choice)>0

    for idx in range(epoch_size):
        
        optimizer.zero_grad()

        dataset = random.choice(choice)
        inp, seg = next(dataloader_iterators[dataset])
        
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
        print ("Epoch %d, Batch %d/%d: Loss=%0.6f"%(epoch, idx+1, epoch_size, avg_tool.get_average()), flush=True)
        loss.backward()
        optimizer.step()

        clip_grad_norm_(net.parameters(), 5.0)

def validate(choice, losstype, num_patients=20):
    val_loss_avg = CumulativeAverager()
    data_strs = ['lgg_t1', 'lgg_t2', 'lgg_flair', 'hgg_t1', 'hgg_t2', 'hgg_flair']
    
    with torch.no_grad():
        rand = random.choice(choice)
        
        for inp, seg in val_dataset_loaders[rand]:
                
            out = cpu_net.forward(inp)
            if losstype == 'bce':
                loss = bce_criterion(out, seg)
            else:
                loss = dice_loss(seg, out)
            val_loss_avg.update(loss)
    
    val_loss = val_loss_avg.get_average()
    print ('Validation Loss on %s set=%0.6f'%(data_strs[rand], val_loss), flush=True)

    return val_loss

def test():
    pass

def saver_fn(net_params, is_best, name='checkpt.pth.tar'):
    torch.save(net_params, name)
		
    if is_best is not None:
        shutil.copyfile(name, 'checkpt_best_%d.pth.tar'%(is_best))

# Train or Test 

# Choices of datasets: can be any subset of the following types of MRI
# [LGG: t1->0, t2->1, flair->2; HGG: t1->3, t2->4, flair->5]

if not args.demo:
    
    avg_tool = CumulativeAverager()
    
    choice, losstype = [0,1,2,3,4,5], 'bce'

    dataloader_iterators = [iter(x) for x in primary_dataset_loaders]
    
    vloss, is_best = torch.tensor(float(np.inf)), None
    for epoch in range(args.epochs):
        train(epoch, choice=choice, losstype=losstype)
        val_loss = validate(choice=choice, losstype=losstype)
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
        
        saver_fn(net, epoch)
else:
    test()


