import shutil

import torch
import os

from utils.dataset import SegmentationDataset


def calculate_class_sizes(train_data):
    class_1 = 0
    background = 0
    for (img, target, _) in train_data:
        class_1_counts = torch.count_nonzero(target)
        class_1 = class_1 + class_1_counts
        background = background + torch.numel(target) - class_1_counts
    return class_1, background


def calculate_mean_and_std(train_data):
    size = len(train_data)
    meanTotal = 0
    stdTotal = 0
    for (img, target, img_id) in train_data:
        mean, std = torch.std_mean(img)
        meanTotal = meanTotal + mean
        stdTotal = stdTotal + std

    return meanTotal/size, stdTotal/size


def copy_file(source, destination):
    shutil.copyfile(source, destination)


def copy_directory(source, destination):
    shutil.copytree(source, destination)

def get_file_containing_word(directory, name):
    dir = os.listdir(directory)
    for fname in dir:
        if name in fname:
            return fname