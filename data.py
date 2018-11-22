import nibabel as nib
import pydicom

import scipy
import glob
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import os

from torch.utils.data import Dataset, DataLoader

from torchvision import transforms 

from medicaltorch import datasets as mt_datasets
from medicaltorch import transforms as mt_transforms
from medicaltorch import losses as mt_losses
from medicaltorch import metrics as mt_metrics
from medicaltorch import filters as mt_filters

from transforms3d import *

class MRIDataset(Dataset):
    
    def __init__(self, tcia_folder=None, brats_folder=None, lgg_folder=None, hgg_folder=None, 
                    type_str='T1', stage='lgg', flag_3d=False, mode='train', channel_size_3d=32, mri_slice_dim=128):
        
        assert stage in ['lgg', 'hgg']
        assert type_str in ['T1', 'T2', 'FLAIR']

        Dataset.__init__(self)
        
        self.flag_3d = flag_3d

        self.tcia_folder = tcia_folder
        self.brats_folder = brats_folder
        self.lgg_folder = lgg_folder
        self.hgg_folder = hgg_folder

        if stage == 'lgg':
            self.folders = glob.glob(self.tcia_folder+'/*')
            self.dataset_types = ['tcia' for _ in range(len(self.folders))]

            brats = glob.glob(self.brats_folder+'/LGG/*')
            self.folders.extend(brats)
            self.dataset_types.extend(['brats' for _ in range(len(brats))])

            lgg = [x for x in glob.glob(self.lgg_folder+'/*') if os.path.isdir(x)]
            self.folders.extend(lgg)
            self.dataset_types.extend(['lgg' for _ in range(len(lgg))])
        else:
            self.folders = glob.glob(self.brats_folder+'/HGG/*')
            self.dataset_types = ['brats' for _ in range(len(self.folders))]
            
            hgg = [x for x in glob.glob(self.hgg_folder+'/*') if os.path.isdir(x)]
            self.folders.extend(hgg)
            self.dataset_types.extend(['hgg' for _ in range(len(hgg))])
        
        self.type = type_str
        self.stage = stage
        self.channel_size_3d = channel_size_3d
        
        self.seg_mapping = {'tcia': 'Segmentation', 'brats':'seg', 
                        'lgg':['GlistrBoost_ManuallyCorrected', 'GlistrBoost*'], 
                        'hgg':['GlistrBoost_ManuallyCorrected', 'GlistrBoost*']}
        
        self.type_mapping = {'tcia': self.type+'*', 'brats': self.type.lower(), 'lgg': self.type.lower(), 'hgg': self.type.lower()}

        self.segmentation_pairs = []
        for idx in range(len(self.folders)):
            
            if isinstance(self.seg_mapping[self.dataset_types[idx]], list):
                try:
                    seg_fname = glob.glob(self.folders[idx] + '/*'+self.seg_mapping[self.dataset_types[idx]][0]+'.nii.gz')[0]
                except Exception:
                    seg_fname = glob.glob(self.folders[idx] + '/*'+self.seg_mapping[self.dataset_types[idx]][1]+'.nii.gz')[0]
            else:
                seg_fname = glob.glob(self.folders[idx] + '/*'+self.seg_mapping[self.dataset_types[idx]]+'.nii.gz')[0]
               
            vox_fname_list = glob.glob(self.folders[idx] + '/*'+self.type_mapping[self.dataset_types[idx]]+'.nii.gz')
            
            if vox_fname_list == []:
                continue
            else:
                vox_fname = vox_fname_list[0]

            self.segmentation_pairs.append([vox_fname, seg_fname])
        
        spl = [.8,.1,.1]
        
        train_ptr = int(spl[0]*len(self.segmentation_pairs)) 
        val_ptr = train_ptr + int(spl[1]*len(self.segmentation_pairs))
 
        if not flag_3d:
            train_transforms = transforms.Compose([
                                MTResize((mri_slice_dim,mri_slice_dim)),
                                mt_transforms.ToTensor(),
                                MTNormalize()])

            val_transforms = transforms.Compose([
                                transforms.Resize((mri_slice_dim,mri_slice_dim)),
                                mt_transforms.ToTensor(),
                                MTNormalize()])       
            train_unnormalized = train_transforms

        else:
            train_transforms = transforms.Compose([
                                    ToPILImage3D(),
                                    Resize3D((mri_slice_dim, mri_slice_dim)),
                                    transforms.RandomChoice([
                                        RandomHorizontalFlip3D(),
                                        RandomVerticalFlip3D(),
                                        RandomRotation3D(30)]),
                                    ToTensor3D(),
                                    Normalize3D('min_max')])

            train_unnormalized = transforms.Compose([
                                    ToPILImage3D(),
                                    Resize3D((mri_slice_dim, mri_slice_dim)),
                                    transforms.RandomChoice([
                                        RandomHorizontalFlip3D(),
                                        RandomVerticalFlip3D(),
                                        RandomRotation3D(30)]),
                                    ToTensor3D(),])
            
            val_transforms = transforms.Compose([ToTensor3D(), IndividualNormalize3D()])

        if mode == 'train':
            self.segmentation_pairs = self.segmentation_pairs[:train_ptr]
            self.transforms = train_transforms
            self.seg_transforms = train_unnormalized
        elif mode == 'val':
            self.segmentation_pairs = self.segmentation_pairs[train_ptr:val_ptr]
            self.transforms = val_transforms
            self.seg_transforms = train_unnormalized
        else:
            self.segmentation_pairs = self.segmentation_pairs[val_ptr:]
            self.transforms = val_transforms
            self.seg_transforms = train_unnormalized
        
        if not flag_3d:
            self.twod_slices_dataset = mt_datasets.MRI2DSegmentationDataset(self.segmentation_pairs, transform=self.transforms)
        
    def __len__(self):
        if not self.flag_3d:
            return len(self.twod_slices_dataset)
        else:
            return len(self.segmentation_pairs)

    def __getitem__(self, idx):
        
        if not self.flag_3d:
            mt_dict = self.twod_slices_dataset.__getitem__(idx)
            return mt_dict['input'], mt_dict['gt']

        else:
            vox_fname, seg_fname = self.segmentation_pairs[idx]
            fobj = nib.load(vox_fname)
            sobj = nib.load(seg_fname)
            inp, out = torch.tensor(fobj.get_fdata()), torch.tensor(sobj.get_fdata())     
            
            inp = inp.permute(2, 0, 1)
            out = out.permute(2, 0, 1)
            
            if inp.size(0)<=self.channel_size_3d:
                batch_size = (self.channel_size_3d,) + inp.size()[1:]
                temp1, temp2 = torch.zeros(batch_size), torch.zeros(batch_size)
                temp1[:inp.size(0),:,:] = inp
                temp2[:out.size(0),:,:] = out

                inp, out = temp1, temp2
                
            else:
                r = np.random.randint(0, inp.size(0)-self.channel_size_3d)
                inp, out = inp[r:r+self.channel_size_3d,:,:], out[r:r+self.channel_size_3d,:,:]
            
            if self.transforms:
                inp, out = self.transforms(inp), self.seg_transforms(out) 
            
            out[out>0] = 1

            return inp, out

    def show_slices(self, slices, save=None):
    
        num_rows = len(slices)
    
        fig, axes = plt.subplots(len(slices), len(slices[0]))
        for i, mri_grp in enumerate(slices):
            for j, img in enumerate(mri_grp):
                if num_rows == 1:
                    axes[j].imshow(img.T, cmap="gray", origin="lower")
                    axes[j].axis('off')
                else:
                    axes[i, j].imshow(img.T, cmap="gray", origin="lower")
                    axes[i, j].axis('off')
        if not save:
            plt.show()
        else:
            plt.savefig(save)

def plot_histogram(t1lgg, t1hgg, t2lgg, t2hgg): 
    
    labels = list(set(t1lgg.dataset_types+t1hgg.dataset_types))
    
    data = []
  
    x_t1lgg = [labels.index(x) for x in t1lgg.dataset_types]
    x_t2lgg = [labels.index(x) for x in t2lgg.dataset_types]
    x_t1hgg = [labels.index(x) for x in t1hgg.dataset_types]
    x_t2hgg = [labels.index(x) for x in t2hgg.dataset_types]
    
    xaxis = [x_t1lgg, x_t2lgg, x_t1hgg, x_t2hgg]
    
    x_out = []
    for x in xaxis:
        keys, ctx = np.unique(x, return_counts=True)
        
        dataset = np.zeros(len(labels))
        for k, ct in zip(keys, ctx):
            dataset[k] = ct

        data.append(dataset)

    data = np.array(data).T

    X = [x for x in range(4)]
    
    fig = plt.figure(figsize=(10,8))
    plt.style.use('seaborn-deep')
    
    colors, cumulative = [(1,0,0,.5),(0,1,0,.5),(0,0,1,.5),(0,0,0,.5)], data[0]
    for i, (d, c) in enumerate(zip(data, colors)):
        if i==0:
            b = plt.bar(X, d, color=c, width=0.5)
        else:
            b = plt.bar(X, d, color=c, bottom=cumulative, width=0.5)
            cumulative += d

    plt.legend(labels, loc='upper right')
    plt.xticks(np.arange(len(labels)), ['LGG T1 MRI', 'LGG T2 MRI', 'HGG T1 MRI', 'HGG T2 MRI'], rotation=30)
    plt.yticks(np.arange(0, 350, 50))
    plt.title('Distribution over Public MR datasets')
    plt.tight_layout()
    
    #plt.show()
    plt.savefig('dataset_dist.png')

if __name__=='__main__':
    
    params = {'tcia_folder':'data/NiFTiSegmentationsEdited', 'brats_folder':'data/BraTs', 
                'hgg_folder':'data/Pre-operative_TCGA_GBM_NIfTI_and_Segmentations',
                'lgg_folder':'data/Pre-operative_TCGA_LGG_NIfTI_and_Segmentations'}
   
    flag_3d = False
    mri_slice_dim = 128

    t1_lgg = MRIDataset(**params, type_str='T1', stage='lgg', flag_3d=flag_3d, mri_slice_dim=mri_slice_dim)
    t2_lgg = MRIDataset(**params, type_str='T2', stage='lgg', flag_3d=flag_3d, mri_slice_dim=mri_slice_dim)
    t1_hgg = MRIDataset(**params, type_str='T1', stage='hgg', flag_3d=flag_3d, mri_slice_dim=mri_slice_dim)
    t2_hgg = MRIDataset(**params, type_str='T2', stage='hgg', flag_3d=flag_3d, mri_slice_dim=mri_slice_dim)
    
    prep = lambda x: 'Number of %s in %s study for %s: %d\n'%('patients' if flag_3d else 'slices', x.type, x.stage, len(x))
    print (prep(t1_lgg), prep(t2_lgg), prep(t1_hgg), prep(t2_hgg))
    
    #plot_histogram(t1_lgg, t1_hgg, t2_lgg, t2_hgg)

    load_t1_lgg = DataLoader(t1_lgg, batch_size=32, shuffle=True, 
                                num_workers=4, collate_fn=mt_datasets.mt_collate)
    '''
    for i, (i1, i2) in enumerate(load_t1_lgg):
        print (i1.shape, i2.shape, [torch.min(i1), torch.max(i1)], [torch.min(i2), torch.max(i2)])
    '''

    flag_3d = True
    channel_size_3d = 32
    t1_lgg_3d = MRIDataset(**params, type_str='T1', stage='lgg', flag_3d=flag_3d, 
                                channel_size_3d=channel_size_3d, mri_slice_dim=mri_slice_dim)
    t2_lgg_3d = MRIDataset(**params, type_str='T2', stage='lgg', flag_3d=flag_3d, 
                                channel_size_3d=channel_size_3d, mri_slice_dim=mri_slice_dim)
    t1_hgg_3d = MRIDataset(**params, type_str='T1', stage='hgg', flag_3d=flag_3d, 
                                channel_size_3d=channel_size_3d, mri_slice_dim=mri_slice_dim)
    t2_hgg_3d = MRIDataset(**params, type_str='T2', stage='hgg', flag_3d=flag_3d, 
                                channel_size_3d=channel_size_3d, mri_slice_dim=mri_slice_dim)
    
    print (prep(t1_lgg_3d), prep(t2_lgg_3d), prep(t1_hgg_3d), prep(t2_hgg_3d))
    
    load_t1_lgg_3d = DataLoader(t1_lgg_3d, batch_size=1, shuffle=False, num_workers=4, collate_fn=mt_datasets.mt_collate)
        
    num_patients, num_imgs = 12, 4

    for i, (i1, i2) in enumerate(load_t1_lgg_3d):
        
        print (i1.shape, i2.shape, [torch.min(i1), torch.max(i1)], [torch.min(i2), torch.max(i2)]) 
        
        nimgs = int(num_patients/num_imgs)
        fig, ax = plt.subplots(nrows=nimgs, ncols=2)
        axptr = 0
        for j in range(i1.size(1)):
            if torch.max(i2[0][j])>0:
                img1 = i1[0][j].permute(1,2,0).squeeze(2)
                img2 = i2[0][j].permute(1,2,0).squeeze(2)
                ax[axptr, 0].imshow(img1)
                ax[axptr, 1].imshow(img2)
                ax[axptr, 0].axis('off')
                ax[axptr, 1].axis('off')
                axptr += 1
            if axptr == nimgs:
                print ("Saved 1 batch image")
                plt.savefig('normalized_'+str(i)+'.png')
                break

    '''
    for i, (i1, i2) in enumerate(t2_lgg):
        print (t1_lgg.dataset_types[i], i1.shape, i2.shape, [np.atorch.min(i1), np.atorch.max(i1)], [np.atorch.min(i2), np.atorch.max(i2)])
    for i, (i1, i2) in enumerate(t1_hgg):
        print (t1_lgg.dataset_types[i], i1.shape, i2.shape, [np.atorch.min(i1), np.atorch.max(i1)], [np.atorch.min(i2), np.atorch.max(i2)])
    for i, (i1, i2) in enumerate(t2_hgg):
        print (t1_lgg.dataset_types[i], i1.shape, i2.shape, [np.atorch.min(i1), np.atorch.max(i1)], [np.atorch.min(i2), np.atorch.max(i2)])
    
    # Save sample dataset image slices to disk
    no_of_patients = 10
    
    TOTAL = []

    rno = -1
    for num in range(no_of_patients):

        rno += 1
        f = folders[rno]

        t1_img = get_tcia_data(f, typ='T1')
        t2_img = get_tcia_data(f, typ='T2')
        seg_img = get_tcia_data(f, typ='Segmentation')
        
        for i in range(seg_img.shape[2]):
            if np.sum(seg_img[:,:,i])>0:
                slices = [[t1_img[:,:,i], t2_img[:,:,i], seg_img[:,:,i]]]
        #show_slices(slices)
        
        TOTAL.extend(slices)
    '''
