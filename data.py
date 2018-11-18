import nibabel as nib
import pydicom

import glob
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import os

from torch.utils.data import Dataset

MRI_Types = {'T1': ['T1', 'SPGR', 'BRAVO', 'Bravo', 't1'], 
                'T2': ['T2', 'FSE', 't2', 'K2']}

class MRIDataset(Dataset):
    
    def __init__(self, tcia_folder=None, brats_folder=None, lgg_folder=None, hgg_folder=None, type_str='T1', stage='lgg'):
        
        assert stage in ['lgg', 'hgg']
        assert type_str in ['T1', 'T2']

        Dataset.__init__(self)
        
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
        
        self.seg_mapping = {'tcia': 'Segmentation', 'brats':'seg', 
                        'lgg':'GlistrBoost_ManuallyCorrected', 
                        'hgg':'GlistrBoost_ManuallyCorrected'}
        
        self.type_mapping = {'tcia': self.type+'*', 'brats': self.type.lower(), 'lgg': self.type.lower(), 'hgg': self.type.lower()}

    def __len__(self):
        return len(self.folders)

    def __getitem__(self, idx):
        
        vox_fname = glob.glob(self.folders[idx] + '/*'+self.type_mapping[self.dataset_types[idx]]+'.nii.gz')[0]
        seg_fname = glob.glob(self.folders[idx] + '/*'+self.seg_mapping[self.dataset_types[idx]]+'.nii.gz')[0]

        fobj = nib.load(vox_fname)
        sobj = nib.load(seg_fname)
        return fobj.get_fdata(), sobj.get_fdata()         

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
    
    t1_lgg = MRIDataset(**params, type_str='T1', stage='lgg')
    t2_lgg = MRIDataset(**params, type_str='T2', stage='lgg')
    t1_hgg = MRIDataset(**params, type_str='T1', stage='hgg')
    t2_hgg = MRIDataset(**params, type_str='T2', stage='hgg')
    
    prep = lambda x: 'Number of patients in %s study for %s: %d\n'%(x.type, x.stage, len(x))
    print (prep(t1_lgg), prep(t2_lgg), prep(t1_hgg), prep(t2_hgg))
    
    plot_histogram(t1_lgg, t1_hgg, t2_lgg, t2_hgg)

    '''
    for i, (i1, i2) in enumerate(t1_lgg):
        print (t1_lgg.dataset_types[i], i1.shape, i2.shape, [np.amin(i1), np.amax(i1)], [np.amin(i2), np.amax(i2)])
    for i, (i1, i2) in enumerate(t2_lgg):
        print (t1_lgg.dataset_types[i], i1.shape, i2.shape, [np.amin(i1), np.amax(i1)], [np.amin(i2), np.amax(i2)])
    for i, (i1, i2) in enumerate(t1_hgg):
        print (t1_lgg.dataset_types[i], i1.shape, i2.shape, [np.amin(i1), np.amax(i1)], [np.amin(i2), np.amax(i2)])
    for i, (i1, i2) in enumerate(t2_hgg):
        print (t1_lgg.dataset_types[i], i1.shape, i2.shape, [np.amin(i1), np.amax(i1)], [np.amin(i2), np.amax(i2)])
    '''

    '''
    # Save sample dataset image slices to disk
    no_of_patients = 4
    
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
