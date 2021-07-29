import shutil

import torch
import os

from utils.dataset import SegmentationDataset


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