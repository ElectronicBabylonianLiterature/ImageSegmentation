import os
import shutil

import torch
from matplotlib import pyplot as plt

from utils.dataset import SegmentationDataset


def log_images(logger, train_data):
    num_example_imgs = 4
    for i, (img, target) in enumerate(train_data[:num_example_imgs]):
        logger.experiment.add_image(img[0], "mask/train", i)
        logger.experiment.add_image(target[0], "mask/mask", i)

def displayImagesTarget(train_data):
    img, target = train_data[0]
    plt.imshow(target.numpy().transpose(1, 2, 0), cmap="gray")
    plt.savefig('target.png', dpi=300)
    plt.show()


def displayImages(train_data, num_example_imgs=4):
    plt.figure(dpi=1200)
    for i, (img, target) in enumerate(train_data[:num_example_imgs]):
        # img
        plt.subplot(num_example_imgs, 2, i * 2 + 1)
        plt.imshow(img.numpy().transpose(1, 2, 0), cmap="gray")
        plt.axis('off')
        if i == 0:
            plt.title("Input image")

        # target
        plt.subplot(num_example_imgs, 2, i * 2 + 2)
        plt.imshow(target.numpy().transpose(1,2,0), cmap="gray")
        plt.axis('off')
        if i == 0:
            plt.title("Target image")
    plt.show()

def displayImagesTest(model, data, num_example_imgs=1):
    plt.figure(dpi=1200)
    model.eval()
    for i, (img, target) in enumerate(data[:num_example_imgs]):
        # img
        plt.subplot(num_example_imgs, 2, i * 2 + 1)
        img = torch.unsqueeze(img, 0)
        img = model.predict_step(img, i)
        plt.imshow(img[0].numpy().transpose(1, 2, 0), cmap="gray")
        plt.axis('off')
        if i == 0:
            plt.title("Input image")

        # target
        plt.subplot(num_example_imgs, 2, i * 2 + 2)
        plt.imshow(target.numpy().transpose(1,2,0), cmap="gray")
        plt.axis('off')
        if i == 0:
            plt.title("Target image")
    plt.savefig('1.png', dpi=300)
    plt.show()


def binarization_threshold_values(image_paths_file, resize, threshold_folder, transform, normalize):
    if not os.path.exists(threshold_folder):
        os.mkdir(threshold_folder)
    #thresholds = np.linspace(0.005 ,0.2, 10, False)
    thresholds = [0.1]
    num_example_imgs=5
    for i in thresholds:
        train_data = SegmentationDataset(image_paths_file=image_paths_file, resize=resize, binarization_threshold=i, transform=transform, normalize=normalize)
        path_for_img = f"{threshold_folder}/threshold-{i}"
        if os.path.exists(path_for_img):
            shutil.rmtree(path_for_img)

        os.mkdir(path_for_img)
        for counter, (img, target, img_id) in enumerate(train_data[:num_example_imgs]):
            saveImageFromTensor(img, img_id, f"{path_for_img}/{img_id}.png")
            saveImageFromTensor(target, img_id, f"{path_for_img}/{img_id}_GT0.png")
            print("Counter", counter)
        print("Threshold", i)


def saveImageFromTensor(img, img_id, path="."):
    saveImage(img.numpy().transpose(1,2,0), img_id, path)


def saveImage(img, img_id, path="."):
    plt.figure(dpi=1200)
    plt.imshow(img, cmap="gray")
    plt.title(img_id)
    plt.savefig(f"{path}/{img_id}.png", dpi=800)


