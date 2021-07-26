import os

import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import SegmentationDataset
from network.unet_model import UNet

root = os.path.dirname(os.path.abspath(__file__))
data_root = os.path.join(root, 'segmentation_data')

resize = (640, 480)
test_data = SegmentationDataset(image_paths_file=f"{data_root}/test.txt", resize=resize)
test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

hparams = {
    "epochs": 1,
    "lr": 0.001
}
model = UNet(1, 1, hparams["lr"])
model.load_state_dict(torch.load("./models/u-net.pt"))
model.eval()

trainer = pl.Trainer(gpus=1, max_epochs=hparams["epochs"])
trainer.test(model, test_loader)




