#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import random
import numpy as np
from glob import glob
from torchvision.datasets import VisionDataset
from tqdm import tqdm
from PIL import Image
from torchvision.datasets.utils import download_and_extract_archive, download_url, extract_archive

import sys
import pdb
import torch

class ForkedPdb(pdb.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child

    """
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin
            
class Kaist(VisionDataset):

    def __init__(self, root, split='train', transforms=None, paired=True, use_thermal=True):

        """
        Kaist Multispectral Pedestrian Detection Benchmark: https://github.com/SoonminHwang/rgbt-ped-detection

        """
        super().__init__(root, transforms)

        self.use_thermal = use_thermal

        self.urls = [
            "https://nexus.anotherbrain.lan/repository/raw-vae-dataset/datasets/kaist/annotations.zip",
            "https://nexus.anotherbrain.lan/repository/raw-vae-dataset/datasets/kaist/imageSets.zip",
            "https://nexus.anotherbrain.lan/repository/raw-vae-dataset/datasets/kaist/set00_V000.zip",
            "https://nexus.anotherbrain.lan/repository/raw-vae-dataset/datasets/kaist/set00_V001.zip",
            "https://nexus.anotherbrain.lan/repository/raw-vae-dataset/datasets/kaist/set00_V002.zip",
            "https://nexus.anotherbrain.lan/repository/raw-vae-dataset/datasets/kaist/set00_V003.zip",
            "https://nexus.anotherbrain.lan/repository/raw-vae-dataset/datasets/kaist/set00_V004.zip",
            "https://nexus.anotherbrain.lan/repository/raw-vae-dataset/datasets/kaist/set00_V005.zip",
            "https://nexus.anotherbrain.lan/repository/raw-vae-dataset/datasets/kaist/set00_V006.zip",
            "https://nexus.anotherbrain.lan/repository/raw-vae-dataset/datasets/kaist/set00_V007.zip",
            "https://nexus.anotherbrain.lan/repository/raw-vae-dataset/datasets/kaist/set00_V008.zip",
            "https://nexus.anotherbrain.lan/repository/raw-vae-dataset/datasets/kaist/set01_V000.zip",
            "https://nexus.anotherbrain.lan/repository/raw-vae-dataset/datasets/kaist/set01_V001.zip",
            "https://nexus.anotherbrain.lan/repository/raw-vae-dataset/datasets/kaist/set01_V002.zip",
            "https://nexus.anotherbrain.lan/repository/raw-vae-dataset/datasets/kaist/set01_V003.zip",
            "https://nexus.anotherbrain.lan/repository/raw-vae-dataset/datasets/kaist/set01_V004.zip",
            "https://nexus.anotherbrain.lan/repository/raw-vae-dataset/datasets/kaist/set01_V005.zip",
            "https://nexus.anotherbrain.lan/repository/raw-vae-dataset/datasets/kaist/set02_V000.zip",
            "https://nexus.anotherbrain.lan/repository/raw-vae-dataset/datasets/kaist/set02_V001.zip",
            "https://nexus.anotherbrain.lan/repository/raw-vae-dataset/datasets/kaist/set02_V002.zip",
            "https://nexus.anotherbrain.lan/repository/raw-vae-dataset/datasets/kaist/set02_V003.zip",
            "https://nexus.anotherbrain.lan/repository/raw-vae-dataset/datasets/kaist/set02_V004.zip",
            "https://nexus.anotherbrain.lan/repository/raw-vae-dataset/datasets/kaist/set03_V000.zip",
            "https://nexus.anotherbrain.lan/repository/raw-vae-dataset/datasets/kaist/set03_V001.zip",
            "https://nexus.anotherbrain.lan/repository/raw-vae-dataset/datasets/kaist/set04_V000.zip",
            "https://nexus.anotherbrain.lan/repository/raw-vae-dataset/datasets/kaist/set04_V001.zip",
            "https://nexus.anotherbrain.lan/repository/raw-vae-dataset/datasets/kaist/set05_V000.zip",
            "https://nexus.anotherbrain.lan/repository/raw-vae-dataset/datasets/kaist/set06_V000.zip",
            "https://nexus.anotherbrain.lan/repository/raw-vae-dataset/datasets/kaist/set06_V001.zip",
            "https://nexus.anotherbrain.lan/repository/raw-vae-dataset/datasets/kaist/set06_V002.zip",
            "https://nexus.anotherbrain.lan/repository/raw-vae-dataset/datasets/kaist/set06_V003.zip",
            "https://nexus.anotherbrain.lan/repository/raw-vae-dataset/datasets/kaist/set06_V004.zip",
            "https://nexus.anotherbrain.lan/repository/raw-vae-dataset/datasets/kaist/set07_V000.zip",
            "https://nexus.anotherbrain.lan/repository/raw-vae-dataset/datasets/kaist/set07_V001.zip",
            "https://nexus.anotherbrain.lan/repository/raw-vae-dataset/datasets/kaist/set07_V002.zip",
            "https://nexus.anotherbrain.lan/repository/raw-vae-dataset/datasets/kaist/set08_V000.zip",
            "https://nexus.anotherbrain.lan/repository/raw-vae-dataset/datasets/kaist/set08_V001.zip",
            "https://nexus.anotherbrain.lan/repository/raw-vae-dataset/datasets/kaist/set08_V002.zip",
            "https://nexus.anotherbrain.lan/repository/raw-vae-dataset/datasets/kaist/set09_V000.zip",
            "https://nexus.anotherbrain.lan/repository/raw-vae-dataset/datasets/kaist/set10_V000.zip",
            "https://nexus.anotherbrain.lan/repository/raw-vae-dataset/datasets/kaist/set10_V001.zip",
            "https://nexus.anotherbrain.lan/repository/raw-vae-dataset/datasets/kaist/set11_V000.zip",
            "https://nexus.anotherbrain.lan/repository/raw-vae-dataset/datasets/kaist/set11_V001.zip"
            ]
        self.split = split
        assert split in ['train', 'valid']

        #print(glob("{}/images/*/*/visible/*".format(root)))
        #exit()
              
        rgb_images = sorted(glob("{}/images/*/*/visible/*".format(root)))
        t_images = sorted(glob("{}/images/*/*/lwir/*".format(root)))
        labels = sorted(glob("{}/annotations/*/*/*".format(root)))

        total = 95324
        
        if not (len(rgb_images) == len(t_images) == total):
            # 95324 is the total # images present in each folder
            for url in self.urls:
                download_and_extract_archive(url, root, extract_root=None, filename=None, md5=None, remove_finished=True)
        
        rgb_images = sorted(glob("{}/images/*/*/visible/*".format(root)))
        t_images = sorted(glob("{}/images/*/*/lwir/*".format(root)))
        assert len(rgb_images) == len(t_images) == total, "lenght rgb_images: {}, length t_images: {}".format(len(rgb_images), len(t_images))

        if not paired:
            # unpairing examples
            rgb_images = rgb_images[::-1]

        if self.split == 'train':
            self.rgb_images = rgb_images[:int(.8 * len(rgb_images))]
            self.t_images = t_images[:int(.8 * len(t_images))]
            self.labels = labels[:int(.8 * len(t_images))]

        elif self.split == 'valid':
            self.rgb_images = rgb_images[int(.8 * len(rgb_images)):]
            self.t_images = t_images[int(.8 * len(t_images)):]
            self.labels = labels[int(.8 * len(t_images)):]

        self.num_samples = len(self.rgb_images)


    def __len__(self):
        return self.num_samples


    def __getitem__(self, index):
        
        image_rgb = Image.open(self.rgb_images[index])
        image_t = Image.open(self.t_images[index])
        #import numpy as np
        #pix = np.array(image_t)
        #import pdb
        #ForkedPdb().set_trace()
        w, h = image_rgb.size
        image_annotations = np.zeros((h, w))

        # TODO here we could get a mask from annotations
        # kaist annotations look like this:
        #person 420 220 66 109 0 0 0 0 0 0 0
        #person 458 219 48 107 0 0 0 0 0 0 0
        with open(self.labels[index]) as fp:
            annotations = fp.readlines()[1:]
        if annotations != []:
            for annotation in annotations:
                cat, x, y, w, h = annotation.split(" ")[:5]
                if cat == "person" or cat == "people":
                    image_annotations[int(y):int(y)+int(h), int(x):int(x)+int(w)] = 1
            #ForkedPdb().set_trace()

        image_annotations = Image.fromarray(np.uint8(image_annotations*255))

        # Mandatory to make pytorch transform involving randomness to work
        seed = np.random.randint(2147483647) # make a seed with numpy generator
        torch.random.manual_seed(seed)
        random.seed(seed) # apply this seed to img tranfsorms
        
        # if self.transforms:

        #     random.seed(seed) # apply this seed to target tranfsorms
        #     image_rgb = self.transforms(image_rgb)

        #     random.seed(seed) # apply this seed to target tranfsorms
        #     image_t = self.transforms(image_t)

        dic = {'img': self.transforms(image_rgb)}

        if self.use_thermal:
            torch.random.manual_seed(seed)
            random.seed(seed) # apply this seed to target tranfsorms
            dic['thermal'] = self.transforms(image_t)[0,:,:].unsqueeze(0)

        # TODO fix this fake mask using annotations
        torch.random.manual_seed(seed)
        random.seed(seed) # apply this seed to target tranfsorms
        dic['masks'] = self.transforms(image_annotations)
        # print("mask size:")
        # print(dic['masks'].shape)
        # print("thermal size:")
        # print(dic['thermal'].shape)
        # exit()
        return dic
        #return image_rgb, image_t


if __name__ == '__main__':

    from torch.utils.data import DataLoader
    from torchvision.transforms import Compose, ToTensor, Resize, Normalize, RandomErasing, Lambda, RandomHorizontalFlip

    pipe = Compose([ Resize((128, 128)), RandomHorizontalFlip(.5), ToTensor()])
    tr_gen = Kaist(root="/home/ofrigo/datasets/kaist/SoonminHwang-rgbt-ped-detection-50ac9b3/data/kaist-rgbt", split='train', paired=False, transforms=pipe)
    te_gen = Kaist(root="/home/ofrigo/datasets/kaist/SoonminHwang-rgbt-ped-detection-50ac9b3/data/kaist-rgbt", split='valid', paired=False, transforms=pipe)
    print(len(tr_gen), len(te_gen))

    loader = DataLoader(te_gen, batch_size=10, num_workers=5)
    import pdb
    pdb.set_trace()
    for a,b in tqdm(loader):
        pass
