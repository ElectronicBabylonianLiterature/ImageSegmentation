import os
import shutil

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from torchvision import transforms

from network.unet_model import UNet
from utils.dataset import SegmentationDataset
from utils.utils import calculate_mean_and_std, copyFile, copyDirectory
from utils.visualization import binarization_threshold_values

root = os.path.dirname(os.path.abspath(__file__))
data_root = os.path.join(root, 'segmentation_data')

logger = TensorBoardLogger("logs", name="u_net", log_graph=True)

resize = transforms.Resize(size=800)
mean, std = calculate_mean_and_std(training_images_path=f"{data_root}/train.txt", transform=resize)


random_affine = transforms.RandomApply([transforms.RandomAffine(degrees=(-20, 20), scale=(1, 2))], 0.5)
transform = transforms.Compose([resize, random_affine])
normalize = transforms.Normalize(mean, std)


hparams = {
    "batchSize": 1,
    "epochs": 1,
    "lr": 0.001,
    "resizeImages": resize,
    "transformations": str(transform) + " Normalize(mean, std)",
    "mean": mean.item(),
    "std": std.item(),
    "binarization_threshold": 0.1,
}


train_data = SegmentationDataset(image_paths_file=f"{data_root}/train.txt", binarization_threshold=hparams["binarization_threshold"], transform=transform, normalize=normalize)
val_data = SegmentationDataset(image_paths_file=f"{data_root}/test.txt", binarization_threshold=hparams["binarization_threshold"], transform=resize, normalize=normalize)
test_data = SegmentationDataset(image_paths_file=f"{data_root}/test.txt", binarization_threshold=hparams["binarization_threshold"], transform=resize, normalize=normalize)


#list(map(lambda batch: print(batch[0].shape), train_data))
#binarization_threshold_values(f"{data_root}/train.txt", "binarization_tests", transform, normalize, len(train_data))

hparams["trainSize"] = len(train_data)
hparams["testSize"] = len(test_data)

print("Train size: %i" % len(train_data))
print("Test size: %i" % len(test_data))
print("Img size: ", train_data[0][0].size())
print("Segmentation size: ", train_data[0][1].size())

def train():
    for counter, index in enumerate(np.random.randint(0, len(train_data) - 1, 10)):
        img, target, img_id = train_data[int(index)]
        logger.experiment.add_image(f"train_samples/sample-{img_id}/training", img)
        logger.experiment.add_image(f"train_samples/sample-{img_id}/truth", target)

    checkpoint_callback = ModelCheckpoint(monitor='Loss/val')
    model = UNet(1, 1, hparams["lr"])
    trainer = pl.Trainer(max_epochs=hparams["epochs"], logger=logger, move_metrics_to_cpu=True, overfit_batches=2)
    logger.log_hyperparams(hparams)
    trainer.fit(model, DataLoader(train_data, batch_size=hparams["batchSize"]))

    #trainer.test(model, DataLoader(val_data, batch_size=1, shuffle=False))
    torch.save(model.state_dict(), f"{logger.log_dir}/model.pt")
    print("Done")


train()


os.mkdir(f"{logger.log_dir}/code")
script_name = os.path.basename(__file__)
copyFile(script_name, f"{logger.log_dir}/code/{script_name}")
copyDirectory("network", f"{logger.log_dir}/code/network")








