import os

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from torchvision import transforms

from utils.CustomTransformers import ElasticTransform
from utils.dataset import SegmentationDataset
from network.unet_model import UNet
from utils.utils import resizeDict, calculate_mean_and_std

root = os.path.dirname(os.path.abspath(__file__))
data_root = os.path.join(root, 'segmentation_data')

logger = TensorBoardLogger("logs", name="u_net", log_graph=True)


resize = (resizeDict(1000, 1), resizeDict(2000, 2), resizeDict(3000, 3), resizeDict(4000, 4),  resizeDict(float("inf"), 5))

mean, std = calculate_mean_and_std(training_images_path=f"{data_root}/train.txt", resize=resize)

randomElastic = transforms.RandomApply([ElasticTransform()], 0.2)
randomAffine = transforms.RandomApply([transforms.RandomAffine(degrees=(-15, 15), scale=(0.7, 1))], 0.3)
transform = transforms.Compose([randomAffine, randomElastic])
normalize = transforms.Normalize(mean, std)


hparams = {
    "batchSize": 1,
    "epochs": 500,
    "lr": 0.001,
    "resizeImages": resize,
    "transformations": str(transform) + " Normalize(mean, std)",
    "mean": mean.item(),
    "std": std.item(),
    "binarization_threshold": 0.1,
}


train_data = SegmentationDataset(image_paths_file=f"{data_root}/train.txt",resize=hparams["resizeImages"], binarization_threshold=hparams["binarization_threshold"], transform=transform, normalize=normalize)
test_data = SegmentationDataset(image_paths_file=f"{data_root}/test.txt", resize=hparams["resizeImages"], binarization_threshold=hparams["binarization_threshold"], normalize=normalize)


hparams["trainSize"] = len(train_data)
hparams["testSize"] = len(test_data)

print("Train size: %i" % len(train_data))
print("Test size: %i" % len(test_data))
print("Img size: ", train_data[0][0].size())
print("Segmentation size: ", train_data[0][1].size())


#binarization_threshold_values(f"{data_root}/binarization.txt", hparams["resizeImages"], "binarization_tests", transform, normalize)

if __name__ == "__main__":
    for counter, index in enumerate(np.random.randint(0, len(train_data) - 1, 10)):
        img, target, img_id = train_data[int(index)]
        logger.experiment.add_image(f"train_samples/sample-{img_id}/training", img)
        logger.experiment.add_image(f"train_samples/sample-{img_id}/truth", target)


    model = UNet(1,1, hparams["lr"])
    trainer = pl.Trainer(gpus=1, max_epochs=hparams["epochs"], logger=logger, precision=16, move_metrics_to_cpu=True)
    logger.log_hyperparams(hparams)
    trainer.fit(model, DataLoader(train_data, batch_size=hparams["batchSize"], num_workers=8))

    trainer.test(model, DataLoader(test_data, batch_size=1, shuffle=False))

    print("Done")

#used when importing into load_model
hparams_for_testing = hparams

