#!/usr/bin/env python3

from trojanvision.datasets.imageset import ImageSet
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets


class TINY(ImageSet):

    name = 'tiny'
    data_shape = [3, 64, 64]
    num_classes = 200

    org_folder_name = {'train': 'tiny/train', 'valid': 'tiny/valid'}

    def initialize(self):
        datasets.ImageFolder(root=self.folder_path, train=True, download=True)
        datasets.ImageFolder(root=self.folder_path, train=False, download=True)

    def _get_org_dataset(self, mode: str, **kwargs):
        assert mode in ['train', 'valid']
        return datasets.ImageFolder(root=self.folder_path+'/train' if mode=='train' \
                                    else self.folder_path + '/valid', **kwargs)
