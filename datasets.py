# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import os
import torch
from torchvision import datasets, transforms
from torch.utils.data.dataset import Dataset

from timm.data.constants import \
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.data import create_transform
import csv
import numpy as np
from PIL import Image

# MIMIC dataloader, adapted from Foundation Ark
class MIMIC(Dataset):

    def __init__(self, images_path, file_path, augment=None, num_class=14,
                 uncertain_label="Ones", unknown_label=0, annotation_percent=100):

        self.img_list = []
        self.img_label = []
        self.augment = augment
        assert uncertain_label in ["Ones", "Zeros", "LSR-Ones", "LSR-Zeros"]
        self.uncertain_label = uncertain_label

        with open(file_path, "r") as fileDescriptor:
            csvReader = csv.reader(fileDescriptor)
            next(csvReader, None)
            for line in csvReader:
                imagePath = os.path.join(images_path, line[0])
                label = line[5:]
                for i in range(num_class):
                    if label[i]:
                        a = float(label[i])
                        if a == 1:
                            label[i] = 1
                        elif a == 0:
                            label[i] = 0
                        elif a == -1:  # uncertain label
                            if self.uncertain_label == "Ones":
                                label[i] = 1
                            elif self.uncertain_label == "Zeros":
                                label[i] = 0
                            elif self.uncertain_label == "LSR-Ones":
                                label[i] = random.uniform(0.55, 0.85)
                            elif self.uncertain_label == "LSR-Zeros":
                                label[i] = random.uniform(0, 0.3)
                    else:
                        label[i] = unknown_label  # unknown label

                self.img_list.append(imagePath)
                self.img_label.append(label)

        indexes = np.arange(len(self.img_list))
        if annotation_percent < 100:
            random.Random(99).shuffle(indexes)
            num_data = int(indexes.shape[0] * annotation_percent / 100.0)
            indexes = indexes[:num_data]

            _img_list, _img_label = copy.deepcopy(self.img_list), copy.deepcopy(self.img_label)
            self.img_list = []
            self.img_label = []

            for i in indexes:
                self.img_list.append(_img_list[i])
                self.img_label.append(_img_label[i])

    def __getitem__(self, index): 

        imagePath = self.img_list[index]
        imageData = Image.open(imagePath).convert('RGB')
        imageLabel = torch.FloatTensor(self.img_label[index])     

        if self.augment is not None: 
            imageData = self.augment(imageData)
        else:
            imageData = (np.array(imageData)).astype('uint8')
            augmented = self.train_augment(image=imageData)
            imageData = augmented['image']
            
            mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            imageData = (imageData - mean) / std
            imageData = imageData.transpose(2, 0, 1).astype('float32')

        return imageData, imageLabel

    def __len__(self):
        return len(self.img_list)

def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    print("Transform = ")
    if isinstance(transform, tuple):
        for trans in transform:
            print(" - - - - - - - - - - ")
            for t in trans.transforms:
                print(t)
    else:
        for t in transform.transforms:
            print(t)
    print("---------------------------")

    if args.data_set == 'CIFAR':
        dataset = datasets.CIFAR100(args.data_path, train=is_train, transform=transform, download=True)
        nb_classes = 100
    elif args.data_set == 'IMNET':
        print("reading from datapath", args.data_path)
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif args.data_set == "MIMIC":
        print("loading MIMIC-CXR ... creating from datapath /scratch/jliang12/data/MIMIC_jpeg/physionet.org/files/mimic-cxr-jpg/2.0.0/")
        img_path = "/scratch/jliang12/data/MIMIC_jpeg/physionet.org/files/mimic-cxr-jpg/2.0.0/"
        if is_train:
            file_path = "/scratch/nralbert/CSE591/Ark/dataset/mimic-cxr-2.0.0-train.csv"
        else:
            file_path = "/scratch/nralbert/CSE591/Ark/dataset/mimic-cxr-2.0.0-test.csv"
        dataset = MIMIC(img_path, file_path, augment=transform)
        nb_classes = 14
    elif args.data_set == "image_folder":
        root = args.data_path if is_train else args.eval_data_path
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = args.nb_classes
        assert len(dataset.class_to_idx) == nb_classes
    else:
        raise NotImplementedError()
    print("Number of the class = %d" % nb_classes)

    return dataset, nb_classes


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    imagenet_default_mean_and_std = args.imagenet_default_mean_and_std
    mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
    std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD

    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        if not resize_im:
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        # warping (no cropping) when evaluated at 384 or larger
        if args.input_size >= 384:  
            t.append(
            transforms.Resize((args.input_size, args.input_size), 
                            interpolation=transforms.InterpolationMode.BICUBIC), 
        )
            print(f"Warping {args.input_size} size input images...")
        else:
            if args.crop_pct is None:
                args.crop_pct = 224 / 256
            size = int(args.input_size / args.crop_pct)
            t.append(
                # to maintain same ratio w.r.t. 224 images
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),  
            )
            t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)
