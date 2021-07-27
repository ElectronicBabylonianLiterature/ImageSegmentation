import os

import pytorch_lightning as pl
import yaml
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from torchvision import transforms

from main import resize
from network.unet_model import UNet
from utils.dataset import SegmentationDataset

unet_version = "version_0"
log_directory = f"logs/u_net/{unet_version}"

checkpoint = f"{log_directory}/checkpoints/epoch=499-step=20999.ckpt"
hparams_yaml = f"{log_directory}/hparams.yaml"

logger = TensorBoardLogger("logs", name=f"u_net_{unet_version}_testing", log_graph=True)
hparams = yaml.load(open(hparams_yaml))

root = os.path.dirname(os.path.abspath(__file__))
data_root = os.path.join(root, 'segmentation_data')

test_data = SegmentationDataset(image_paths_file=f"{data_root}/test.txt", binarization_threshold=hparams["binarization_threshold"], transform=resize, normalize=transforms.Normalize(hparams["mean"], hparams["std"]))
test_loader = DataLoader(test_data, batch_size=1, shuffle=False)


model = UNet.load_from_checkpoint(checkpoint, n_channels=1, n_classes=1, lr=hparams["lr"])

trainer = pl.Trainer(gpus=1, max_epochs=hparams["epochs"], logger=logger, precision=16, move_metrics_to_cpu=True)
trainer.test(model, test_loader)




