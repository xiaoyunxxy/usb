#!/usr/bin/env python3

from trojanvision.datasets.imagefolder import ImageFolder
from trojanzoo.utils.module import Module

from torchvision import datasets
import os
import json
import torchvision.transforms as transforms


from trojanvision import __file__ as root_file
root_dir = os.path.dirname(root_file)


class ImageNet(ImageFolder):
    r"""ImageNet (ILSVRC2012) dataset introduced by Jia Deng and Feifei Li in 2012.
    It inherits :class:`trojanvision.datasets.ImageFolder`.

    See Also:
        * torchvision: :any:`torchvision.datasets.ImageNet`
        * paper: `ImageNet\: A large-scale hierarchical image database`_
        * website: https://image-net.org/about.php

    Note:
        According to https://github.com/pytorch/vision/issues/1563,
        You need to personally visit https://image-net.org/download-images.php
        to download the dataset.

        Expected files:

            * ``'{self.folder_path}/ILSVRC2012_devkit_t12.tar.gz'``
            * ``'{self.folder_path}/ILSVRC2012_img_train.tar'``
            * ``'{self.folder_path}/ILSVRC2012_img_val.tar'``
            * ``'{self.folder_path}/meta.bin'``

    Attributes:
        name (str): ``'imagenet'``
        num_classes (int): ``1000``
        data_shape (list[int]): ``[3, 224, 224]``
        norm_par (dict[str, list[float]]):
            | ``{'mean': [0.485, 0.456, 0.406],``
            | ``'std'  : [0.229, 0.224, 0.225]}``

    .. _ImageNet\: A large-scale hierarchical image database:
        https://ieeexplore.ieee.org/document/5206848
    """

    name = 'imagenet'
    url = {
        'train': 'http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_train.tar',
        'valid': 'http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_val.tar',
        'test': 'http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_test.tar',
    }
    md5 = {
        'train': '1d675b47d978889d74fa0da5fadfb00e',
        'valid': '29b22e2961454d5413ddabcf34fc5622',
        'devkit': 'fa75699e90414af021442c21a62c3abf',
    }

    def __init__(self, norm_par: dict[str, list[float]] = {'mean': [0.485, 0.456, 0.406],
                                                           'std': [0.229, 0.224, 0.225], },
                 **kwargs):
        super().__init__(norm_par=norm_par, **kwargs)

    def initialize_folder(self):
        try:
            datasets.ImageNet(root=self.folder_path, split='train')
            datasets.ImageNet(root=self.folder_path, split='val')
        except RuntimeError:
            raise RuntimeError('\n\n'
                               'You need to visit https://image-net.org/download-images.php '
                               'to download ImageNet.\n'
                               'There are direct links to files, but not legal to distribute. '
                               'Please apply for access permission and find links yourself.\n\n'
                               f'folder_path: {self.folder_path}\n'
                               'expected files:\n'
                               '{folder_path}/ILSVRC2012_devkit_t12.tar.gz\n'
                               '{folder_path}/ILSVRC2012_img_train.tar\n'
                               '{folder_path}/ILSVRC2012_img_val.tar\n'
                               '{folder_path}/meta.bin')
        if not os.path.isdir(os.path.join(self.folder_path, 'valid')):
            os.symlink(os.path.join(self.folder_path, 'val'),
                       os.path.join(self.folder_path, 'valid'))

    def _get_org_dataset(self, mode: str, data_format: str = None,
                         **kwargs) -> datasets.DatasetFolder:
        data_format = data_format or self.data_format
        split = 'val' if mode == 'valid' else mode
        return datasets.ImageNet(root=self.folder_path, split=split, **kwargs)

    def get_class_names(self) -> list[str]:
        if hasattr(self, 'class_names'):
            return getattr(self, 'class_names')
        dataset: datasets.ImageNet = self.get_org_dataset('train')
        classes: list[tuple[str, ...]] = dataset.classes
        return [clss[0] for clss in classes]


class Sample_ImageNet(ImageFolder):

    name: str = 'sample_imagenet'
    data_shape = [3, 224, 224]
    num_classes = 10

    url = {}
    org_folder_name = {'train': 'sample_imagenet/train', 'valid': 'sample_imagenet/val'}


    def _get_org_dataset(self, mode: str, **kwargs):
        assert mode in ['train', 'valid']
        return datasets.ImageFolder(root=self.folder_path+'/train' if mode=='train' \
                                    else self.folder_path + '/val', **kwargs)


    def get_transform(self, mode: str) -> transforms.Compose:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

        if mode == 'train':
            transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize])
        else:
            transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])
        return transform






