import torch
import math
import sys
import random
from PIL import Image
try:
    import accimage
except ImportError:
    accimage = None
import numpy as np
import numbers
import types
import collections
import warnings

from torchvision.transforms import functional as F
from torchvision import transforms

from collections import Iterable

class MTResize(object):
    
    def __init__(self, new_shape, interpolation=Image.BILINEAR, labeled=True):
        
        self.shape = new_shape
        self.interpolation = interpolation
        self.labeled = labeled

    def __call__(self, sample):
        rdict = {}
        input_data = sample['input']
        input_metadata = sample['input_metadata']

        input_data = input_data.resize(self.shape, resample=self.interpolation)
        rdict['input'] = input_data

        if self.labeled:
            gt_data = sample['gt']
            gt_metadata = sample['gt_metadata']
            gt_data = gt_data.resize(self.shape, resample=self.interpolation)
            np_gt_data = np.array(gt_data)
            np_gt_data[np_gt_data >= 0.5] = 1.0
            np_gt_data[np_gt_data < 0.5] = 0.0
            gt_data = Image.fromarray(np_gt_data, mode='F')
            rdict['gt'] = gt_data

        sample.update(rdict)
        return sample

class ToPILImage3D(object):
    
    def __call__(self, pic):
        
        pil_list = []
        for im in pic:
            pil_list.append(Image.fromarray(im.numpy()))
        return pil_list

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        if self.mode is not None:
            format_string += 'mode={0}'.format(self.mode)
        format_string += ')'
        return format_string

class ToTensor3D(object):
    
    def __call__(self, pic):

        for i, im in enumerate(pic):
            pic[i] = F.to_tensor(im)
        return torch.stack(pic)

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        if self.mode is not None:
            format_string += 'mode={0}'.format(self.mode)
        format_string += ')'
        return format_string

class MTNormalize(object):
    
    def __init__(self, proc='min_max'):
    # proc: z_score and min_max
        
        self.proc = proc

    def __call__(self, tensor_dict):
        
        if self.proc == 'z_score':
            mean, std = torch.mean(tensor_dict['input']), torch.std(tensor_dict['input'])
            tensor_dict['input'] -= mean
            tensor_dict['input'] /= std+1e-8
        elif self.proc == 'min_max':
            nmin, nmax = torch.min(tensor_dict['input']), torch.max(tensor_dict['input'])
            inp = (tensor_dict['input'] - nmin) / (nmax-nmin+1e-8)
            tensor_dict['input'] = inp

        return tensor_dict

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class IndividualNormalize3D(object):
    
    def __init__(self, proc='min_max'):
    # proc: z_score and min_max
        
        self.proc = proc

    def __call__(self, blk):
        
        if self.proc == 'z_score':
            for i, b in enumerate(blk):
                mean, std = torch.mean(b), torch.std(b)
                blk[i] = (b - mean)/(std+1e-8)
        elif self.proc == 'min_max':
            for i, b in enumerate(blk):
                nmin, nmax = torch.min(b), torch.max(b)
                blk[i] = (b - nmin) / (nmax-nmin+1e-8)
        blk = blk.squeeze(2)
        return blk

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class Normalize3D(object):
    
    def __init__(self, proc='min_max'):
    # proc: z_score and min_max
        
        self.proc = proc

    def __call__(self, blk):
        
        if self.proc == 'z_score':
            mean, std = torch.mean(blk), torch.std(blk)
            blk = (blk - mean)/(std+1e-8)
        elif self.proc == 'min_max':
            nmin, nmax = torch.min(blk), torch.max(blk)
            blk = (blk - nmin) / (nmax-nmin+1e-8)
        
        return blk

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class Resize3D(object):
    
    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size, int) or (isinstance(size, Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        
        for i, im in enumerate(img):
            img[i] = F.resize(im, self.size, self.interpolation)
        
        return img

    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(self.size, interpolate_str)

class Pad3D(object):

    def __init__(self, padding, fill=0, padding_mode='constant'):
        assert isinstance(padding, (numbers.Number, tuple))
        assert isinstance(fill, (numbers.Number, str, tuple))
        assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric']
        if isinstance(padding, Sequence) and len(padding) not in [2, 4]:
            raise ValueError("Padding must be an int or a 2, or 4 element tuple, not a " +
                             "{} element tuple".format(len(padding)))

        self.padding = padding
        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, img):
        
        for im in img:
            F.pad(im, self.padding, self.fill, self.padding_mode, inplace=True)
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(padding={0}, fill={1}, padding_mode={2})'.\
            format(self.padding, self.fill, self.padding_mode)

class RandomHorizontalFlip3D(object):

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        
        if random.random() < self.p:
            for i, im in enumerate(img):
                img[i] = F.hflip(im)
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

class RandomVerticalFlip3D(object):

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        
        if random.random() < self.p:
            for i, im in enumerate(img):
                img[i] = F.vflip(im)
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

class RandomRotation3D(object):

    def __init__(self, degrees, resample=False, expand=False, center=None):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees

        self.resample = resample
        self.expand = expand
        self.center = center

    @staticmethod
    def get_params(degrees):
        angle = random.uniform(degrees[0], degrees[1])

        return angle

    def __call__(self, img):

        angle = self.get_params(self.degrees)
        
        for i, im in enumerate(img):
            img[i] = F.rotate(im, angle, self.resample, self.expand, self.center)

        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '(degrees={0}'.format(self.degrees)
        format_string += ', resample={0}'.format(self.resample)
        format_string += ', expand={0}'.format(self.expand)
        if self.center is not None:
            format_string += ', center={0}'.format(self.center)
        format_string += ')'
        return format_string


class RandomAffine3D(object):

    def __init__(self, degrees, translate=None, scale=None, shear=None, resample=False, fillcolor=0):
        
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            assert isinstance(degrees, (tuple, list)) and len(degrees) == 2, \
                "degrees should be a list or tuple and it must be of length 2."
            self.degrees = degrees

        if translate is not None:
            assert isinstance(translate, (tuple, list)) and len(translate) == 2, \
                "translate should be a list or tuple and it must be of length 2."
            for t in translate:
                if not (0.0 <= t <= 1.0):
                    raise ValueError("translation values should be between 0 and 1")
        self.translate = translate

        if scale is not None:
            assert isinstance(scale, (tuple, list)) and len(scale) == 2, \
                "scale should be a list or tuple and it must be of length 2."
            for s in scale:
                if s <= 0:
                    raise ValueError("scale values should be positive")
        self.scale = scale

        if shear is not None:
            if isinstance(shear, numbers.Number):
                if shear < 0:
                    raise ValueError("If shear is a single number, it must be positive.")
                self.shear = (-shear, shear)
            else:
                assert isinstance(shear, (tuple, list)) and len(shear) == 2, \
                    "shear should be a list or tuple and it must be of length 2."
                self.shear = shear
        else:
            self.shear = shear

        self.resample = resample
        self.fillcolor = fillcolor

    @staticmethod
    def get_params(degrees, translate, scale_ranges, shears, img_size):
        
        angle = random.uniform(degrees[0], degrees[1])
        if translate is not None:
            max_dx = translate[0] * img_size[0]
            max_dy = translate[1] * img_size[1]
            translations = (np.round(random.uniform(-max_dx, max_dx)),
                            np.round(random.uniform(-max_dy, max_dy)))
        else:
            translations = (0, 0)

        if scale_ranges is not None:
            scale = random.uniform(scale_ranges[0], scale_ranges[1])
        else:
            scale = 1.0

        if shears is not None:
            shear = random.uniform(shears[0], shears[1])
        else:
            shear = 0.0

        return angle, translations, scale, shear

    def __call__(self, img):
        
        ret = self.get_params(self.degrees, self.translate, self.scale, self.shear, img.size)
	    
        for im in img:
            F.affine(im, *ret, resample=self.resample, fillcolor=self.fillcolor, inplace=True)
        
        return img

    def __repr__(self):
        s = '{name}(degrees={degrees}'
        if self.translate is not None:
            s += ', translate={translate}'
        if self.scale is not None:
            s += ', scale={scale}'
        if self.shear is not None:
            s += ', shear={shear}'
        if self.resample > 0:
            s += ', resample={resample}'
        if self.fillcolor != 0:
            s += ', fillcolor={fillcolor}'
        s += ')'
        d = dict(self.__dict__)
        d['resample'] = _pil_interpolation_to_str[d['resample']]
        return s.format(name=self.__class__.__name__, **d)
