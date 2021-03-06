import os

import pytorch_lightning as pl
import yaml
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from torchvision import transforms

from network.unet_model import UNet
from utils.custom_transforms import BinarizationTransform
from utils.dataset import SegmentationDataset
from utils.utils import get_file_containing_word

unet_version = "version_0"
log_directory = f"logs/u_net/{unet_version}"


hparams_yaml = f"{log_directory}/hparams.yaml"

logger = TensorBoardLogger("logs", name=f"u_net_{unet_version}_testing", log_graph=True)
hparams = yaml.load(open(hparams_yaml))

root = os.path.dirname(os.path.abspath(__file__))
data_root = os.path.join(root, 'segmentation_data')

resize=transforms.Resize(size=500)

test_data = SegmentationDataset(image_paths_file=f"{data_root}/test.txt", binarization=BinarizationTransform(hparams["binarization_threshold"]), resize=resize)
test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

checkpoints_directory = f"{log_directory}/checkpoints"
checkpoint = get_file_containing_word(checkpoints_directory, "499")
model = UNet.load_from_checkpoint(f"{checkpoints_directory}/{checkpoint}", n_channels=1, n_classes=1, lr=hparams["lr"])

trainer = pl.Trainer(gpus=1, precision=16, max_epochs=hparams["epochs"], logger=logger)
trainer.test(model, test_loader)




