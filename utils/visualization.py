import os
import shutil
from collections import Sequence

import torch
from matplotlib import pyplot as plt

from utils.custom_transforms import BinarizationTransform
from utils.dataset import SegmentationDataset


def save_images_from_tensors(data, threshold_folder, num_example_imgs=10, binarization_threshold=0.1):
    if not os.path.exists(threshold_folder):
        os.mkdir(threshold_folder)
    threshold = []
    if isinstance(binarization_threshold, float):
        threshold = [binarization_threshold]
    elif isinstance(binarization_threshold, Sequence):
        threshold = binarization_threshold
    elif isinstance(binarization_threshold, BinarizationTransform):
        threshold = [binarization_threshold.threshold]

    for i in threshold:

        path_for_img = f"{threshold_folder}/binarization_threshold_{i}"

        if os.path.exists(path_for_img):
            shutil.rmtree(path_for_img)
        os.mkdir(path_for_img)

        for counter, (img, target, img_id) in enumerate(data[:num_example_imgs]):
            saveImageFromTensor(img,  img_id ,path_for_img, )
            saveImageFromTensor(target, f"{img_id}_GT0", path_for_img, )
            print("Counter", counter)
        print("Threshold", i)


def saveImageFromTensor(img,   img_id, path=".",):
    saveImage(img.numpy().transpose(1,2,0), img_id, path)


def saveImage(img,   img_id, path=".",):
    plt.figure(dpi=1200)
    plt.imshow(img, cmap="gray")
    plt.title(img_id)
    plt.savefig(f"{path}/{img_id}.png", dpi=800)
    plt.close()



