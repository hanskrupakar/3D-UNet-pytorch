from model import UNet3D
from data import MRIDataset
import argparse

folder_params = {'tcia_folder':'data/NiFTiSegmentationsEdited', 'brats_folder':'data/BraTs', 
                    'hgg_folder':'data/Pre-operative_TCGA_GBM_NIfTI_and_Segmentations',
                    'lgg_folder':'data/Pre-operative_TCGA_LGG_NIfTI_and_Segmentations'}
   
ap = argparse.ArgumentParser()
ap.add_argument('--batch_size', type=int, help='Number of 3D voxel batches')
ap.add_argument('--lr', type=float, help='Initial Learning rate')
ap.add_argument('--lr_decay_epochs', type=int, help='Number of initial epochs to delay lr')
ap.add_argument('--lr_decay', type=float, help='Learning Rate Decay')
ap.add_argument('--optimizer', help='sgd, rmsprop, adam')
ap.add_argument('--aug', action='store_true', help='Flag to decide about input augmentations')
args = ap.parse_args()

flag_3d = True
mri_slice_dim = 256

t1_lgg = MRIDataset(**folder_params, type_str='T1', stage='lgg', flag_3d=flag_3d, mri_slice_dim=mri_slice_dim)
t2_lgg = MRIDataset(**folder_params, type_str='T2', stage='lgg', flag_3d=flag_3d, mri_slice_dim=mri_slice_dim)
flair_lgg = MRIDataset(**folder_params, type_str='FLAIR', stage='lgg', flag_3d=flag_3d, mri_slice_dim=mri_slice_dim)

t1_hgg = MRIDataset(**folder_params, type_str='T1', stage='hgg', flag_3d=flag_3d, mri_slice_dim=mri_slice_dim)
t2_hgg = MRIDataset(**folder_params, type_str='T2', stage='hgg', flag_3d=flag_3d, mri_slice_dim=mri_slice_dim)
flair_hgg = MRIDataset(**folder_params, type_str='FLAIR', stage='hgg', flag_3d=flag_3d, mri_slice_dim=mri_slice_dim)
    

