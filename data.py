import nibabel as nib
import glob
import matplotlib.pyplot as plt
import numpy as np

def show_slices(slices):
    fig, axes = plt.subplots(1, len(slices))
    for i, s in enumerate(slices):
        axes[i].imshow(s.T, cmap="gray", origin="lower")
    plt.show()

types = ["T1", "T2", "Segmentation"]

for typ in types:
    
    files = glob.glob("NiFTiSegmentationsEdited/*/*" + typ + "*.nii.gz")
    
    for f in files:
        img_obj = nib.load(f)
        
        if typ=='Segmentation':
            img = img_obj.get_fdata()
            
            l,w,h = img.shape
            
            r1, r2, r3 = np.random.randint(l), np.random.randint(w), np.random.randint(h)
            slices = [img[r1,:,:], img[:,r2,:], img[:,:,r3]]
            show_slices(slices)
