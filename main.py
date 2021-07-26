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
from visualization import displayImages, displayImagesTest, displayImagesTarget

logger = TensorBoardLogger("logs", name="u_net", log_graph=True)

root = os.path.dirname(os.path.abspath(__file__))
data_root = os.path.join(root, 'segmentation_data')


transformations = [transforms.RandomAffine(degrees=(-10, 10))]
random = transforms.RandomApply(transformations, 0.5)

transform = transforms.Compose([random])
resize = (640, 480)
train_data = SegmentationDataset(image_paths_file=f"{data_root}/train.txt",resize=resize, transform=transform)
#val_data = SegmentationDataset(image_paths_file=f"{data_root}/val.txt", resize=resize, transform= transform)
test_data = SegmentationDataset(image_paths_file=f"{data_root}/test.txt", resize=resize, transform=transform)


print("Train size: %i" % len(train_data))
print("Test size: %i" % len(test_data))
print("Img size: ", train_data[0][0].size())
print("Segmentation size: ", train_data[0][1].size())


#displayImages(train_data)
#displayImagesTarget(train_data)

hparams = {
    "epochs": 100,
    "lr": 0.001
}
model = UNet(n_channels=1, n_classes=1, lr=hparams["lr"])

logger.log_hyperparams(hparams)

trainer = pl.Trainer(default_root_dir="/home/yunus/PycharmProjects/aru-net/checkpoints", gpus=1, max_epochs=hparams["epochs"], logger=logger,  log_every_n_steps=1)
trainer.fit(model, DataLoader(train_data, batch_size=2, num_workers=4))

#displayImagesTest(model, test_data)

trainer.test(model, DataLoader(test_data, shuffle=False))




torch.save(model.state_dict(), f"./models/model-1.pt")

print("Done")
