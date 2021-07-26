import os

from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl
import torch
from matplotlib import pyplot as plt
from pytorch_lightning.callbacks import EarlyStopping
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import SegmentationDataset

from network.unet_model import UNet

root = os.path.dirname(os.path.abspath(__file__))
data_root = os.path.join(root, 'segmentation_data')

logger = TensorBoardLogger("logs", name="u_net", log_graph=True)

transformations = [transforms.RandomAffine(degrees=(-10, 10))]
random = transforms.RandomApply(transformations, 0.5)
transform = transforms.Compose([random])
resize = (640, 480)

train_data = SegmentationDataset(image_paths_file=f"{data_root}/train.txt",resize=resize, transform=transform)
test_data = SegmentationDataset(image_paths_file=f"{data_root}/test.txt", resize=resize, transform=transform)

print("Train size: %i" % len(train_data))
print("Test size: %i" % len(test_data))
print("Img size: ", train_data[0][0].size())
print("Segmentation size: ", train_data[0][1].size())


hparams = {
    "epochs": 1,
    "lr": 0.001,
    "resizeImages": resize,
    "trainSize": len(train_data),
    "testSize": len(test_data),
    "transformations": str(transform)
}
logger.log_hyperparams(hparams)

model = UNet(n_channels=1, n_classes=1, lr=hparams["lr"])
logger.log_graph(model)

trainer = pl.Trainer(gpus=1, max_epochs=hparams["epochs"], logger=logger, log_every_n_steps=1)
trainer.fit(model, DataLoader(train_data, batch_size=1, num_workers=4))

trainer.test(model, DataLoader(test_data, batch_size=1, shuffle=False))
torch.save(model.state_dict(), f"./models/u-net.pt")

print("Done")
