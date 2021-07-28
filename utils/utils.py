import shutil

import torch

from utils.dataset import SegmentationDataset


def calculate_mean_and_std(training_images_path, transform):
    train_data = SegmentationDataset(image_paths_file=training_images_path, transform=transform, binarization_threshold=0.1)

    size = len(train_data)
    meanTotal = 0
    stdTotal = 0
    for (img, target, img_id) in train_data:
        mean, std = torch.std_mean(img)
        meanTotal = meanTotal + mean
        stdTotal = stdTotal + std

    return meanTotal/size, stdTotal/size


def copyFile(source, destination):
    shutil.copyfile(source, destination)


def copyDirectory(source, destination):
    shutil.copytree(source, destination)