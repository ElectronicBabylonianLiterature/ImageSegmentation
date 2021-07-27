import os

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from torchvision import transforms

from CustomTransformers import ElasticTransform
from dataset import SegmentationDataset
from network.unet_model import UNet
from visualization import binarization_threshold_values, saveImage

root = os.path.dirname(os.path.abspath(__file__))
data_root = os.path.join(root, 'segmentation_data')

logger = TensorBoardLogger("logs", name="u_net", log_graph=True)

def resizeDict(maxValue, rescale):
    return {"maxValue": maxValue, "rescale": rescale}

def calculate_mean_and_std(training_images_path, resize):
    train_data = SegmentationDataset(image_paths_file=training_images_path, resize=resize, binarization_threshold=0.1)

    size = len(train_data)
    meanTotal = 0
    stdTotal = 0
    for (img, target, img_id) in train_data:
        mean, std = torch.std_mean(img)
        meanTotal = meanTotal + mean
        stdTotal = stdTotal + std

    return meanTotal/size, stdTotal/size



resize = (resizeDict(1000, 1), resizeDict(2000, 1.5), resizeDict(4000, 3),  resizeDict(float("inf"), 4))

mean, std = calculate_mean_and_std(training_images_path=f"{data_root}/train.txt", resize=resize)


randomElastic = transforms.RandomApply([ElasticTransform()], 0.2)
randomAffine = transforms.RandomApply([transforms.RandomAffine(degrees=(-15, 15), scale=(0.7, 1.2))], 0.3)
transform = transforms.Compose([randomAffine, randomElastic])
normalize= transforms.Normalize(mean, std)


hparams = {
    "batchSize": 1,
    "epochs": 350,
    "lr": 0.001,
    "resizeImages": resize,
    "transformations": transform,
    "binarization_threshold": 0.1,
}


train_data = SegmentationDataset(image_paths_file=f"{data_root}/train.txt",resize=hparams["resizeImages"], binarization_threshold=hparams["binarization_threshold"], transform=hparams["transformations"], normalize=transforms.Normalize(mean, std))
test_data = SegmentationDataset(image_paths_file=f"{data_root}/test.txt", resize=hparams["resizeImages"], binarization_threshold=hparams["binarization_threshold"], transform=hparams["transformations"], normalize=transforms.Normalize(mean, std))


hparams["trainSize"] = len(train_data)
hparams["testSize"] = len(test_data)

print("Train size: %i" % len(train_data))
print("Test size: %i" % len(test_data))
print("Img size: ", train_data[0][0].size())
print("Segmentation size: ", train_data[0][1].size())


#binarization_threshold_values(f"{data_root}/binarization.txt", hparams["resizeImages"], "binarization_tests", transform, normalize)



for counter, index in enumerate(np.random.randint(0, len(train_data) - 1, 10)):
    img, target, img_id = train_data[int(index)]
    logger.experiment.add_image(f"train_samples/sample-{img_id}/training", img)
    logger.experiment.add_image(f"train_samples/sample-{img_id}/truth", target)


model = UNet(n_channels=1, n_classes=1, lr=hparams["lr"])


logger.log_graph(model)
trainer = pl.Trainer(gpus=1, max_epochs=hparams["epochs"], logger=logger)
logger.log_hyperparams(hparams)
trainer.fit(model, DataLoader(train_data, batch_size=hparams["batchSize"], num_workers=4))

trainer.test(model, DataLoader(test_data, batch_size=1, shuffle=False))
torch.save(model.state_dict(), f"./models/u-net.pt")

print("Done")


