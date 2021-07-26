import os

import torch
from torch.utils.data import DataLoader

from dataset import SegmentationDataset
from network.unet_model import UNet
from visualization import displayImagesTest

root = os.path.dirname(os.path.abspath(__file__))
data_root = os.path.join(root, 'segmentation_data')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


test_data = SegmentationDataset(image_paths_file=f"{data_root}/test.txt", resize=(640, 480))
test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

hparams = {
    "lr": 0.001
}
model = UNet(1, 1, hparams["lr"])
model.load_state_dict(torch.load("./models/model-1.pt"))
model.eval()

displayImagesTest(model, test_data)



