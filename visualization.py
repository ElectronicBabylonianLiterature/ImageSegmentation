import torch
from matplotlib import pyplot as plt
from torchvision import transforms


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

