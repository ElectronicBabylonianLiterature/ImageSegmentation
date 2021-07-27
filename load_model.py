import os

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms

from utils.dataset import SegmentationDataset
from main import hparams_for_testing, logger
from network.unet_model import UNet

root = os.path.dirname(os.path.abspath(__file__))
data_root = os.path.join(root, 'segmentation_data')

hparams = hparams_for_testing

test_data = SegmentationDataset(image_paths_file=f"{data_root}/test.txt", resize=hparams["resizeImages"], binarization_threshold=hparams["binarization_threshold"], normalize=transforms.Normalize(hparams["mean"], hparams["std"]) )
test_loader = DataLoader(test_data, batch_size=1, shuffle=False)


model = UNet.load_from_checkpoint("logs/u_net/version_0/checkpoints/epoch=499-step=20999.ckpt", n_channels=1, n_classes=1, lr=hparams["lr"])
#model.load_state_dict(torch.load("./models/u-net.pt"))
model.eval()

trainer = pl.Trainer(gpus=1, max_epochs=hparams["epochs"], logger=logger)
trainer.test(model, test_loader)




