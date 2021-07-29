import os

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from torchvision import transforms

from network.unet_model import UNet
from utils.custom_transforms import ElasticTransform, DynamicResize, BinarizationTransform
from utils.dataset import SegmentationDataset
from utils.utils import calculate_mean_and_std, copy_file, copy_directory, calculate_class_sizes
from utils.visualization import save_images_from_tensors

def save_images():
    list(map(lambda batch: print(batch[0].shape), train_data))
    save_images_from_tensors(train_data, "images", 6, binarization)

root = os.path.dirname(os.path.abspath(__file__))
data_root = os.path.join(root, 'segmentation_data')

logger = TensorBoardLogger("logs", name="u_net", log_graph=True)

resize_value = 700
resize = transforms.Resize(size=resize_value)

binarization_threshold = 0.1
binarization = BinarizationTransform(binarization_threshold)

"""
resize_value = ((1000, 1), (2000, 2), (3000, 3), (4000, 4), (float("inf"), 5))
resize = DynamicResize(size=resize_value)
"""

statistics_data_set = SegmentationDataset(image_paths_file=f"{data_root}/train.txt", resize=resize, binarization=binarization)
mean, std = calculate_mean_and_std(statistics_data_set)

random_elastic = transforms.RandomApply([ElasticTransform()], 0.5)
random_affine = transforms.RandomApply([transforms.RandomAffine(degrees=0, scale=(1, 1.75))], 0.5)
transform = transforms.Compose([random_affine])
#normalize = transforms.Normalize(mean, std)


hparams = {
    "batchSize": 1,
    "epochs": 600,
    "lr": 0.001,
    "resizeValue": resize_value,
    "resizeImages": str(resize),
    "transformations": str(transform),
    "mean": mean.item(),
    "std": std.item(),
    "binarization_threshold": binarization_threshold,
    "binarize_after_resizing": True if binarization is not None else False,
    "accumulateGradients": 8,
}

train_data = SegmentationDataset(image_paths_file=f"{data_root}/train.txt", resize=resize, binarization=binarization, transform=transform)
val_data = SegmentationDataset(image_paths_file=f"{data_root}/test.txt", resize=resize, binarization=binarization)
test_data = SegmentationDataset(image_paths_file=f"{data_root}/test.txt", resize=resize, binarization=binarization)




class_1, background = calculate_class_sizes(train_data)
hparams["lossWeight"] =  int(background/class_1)

hparams["trainSize"] = len(train_data)
hparams["testSize"] = len(test_data)

print("Train size: %i" % len(train_data))
print("Test size: %i" % len(test_data))
print("Img size: ", train_data[0][0].size())
print("Segmentation size: ", train_data[0][1].size())


def copyCodeToLogs():
    os.mkdir(f"{logger.log_dir}/code")
    script_name = os.path.basename(__file__)
    copy_file(script_name, f"{logger.log_dir}/code/{script_name}")
    copy_directory("network", f"{logger.log_dir}/code/network")


def logSomeTrainingImages():
    for counter, index in enumerate(np.random.randint(0, len(train_data) - 1, 10)):
        img, target, img_id = train_data[int(index)]
        logger.experiment.add_image(f"train_samples/sample-{img_id}/training", img)
        logger.experiment.add_image(f"train_samples/sample-{img_id}/truth", target)


if __name__ == '__main__':
    #save_images()
    logSomeTrainingImages()
    copyCodeToLogs()
    trainer = None
    model = None
    try:
        checkpoint_loss = ModelCheckpoint(monitor='Loss/val',
                                          filename="best_val_loss-epoch={epoch:02d}-val_loss=val_loss{val/loss:.2f}",
                                          auto_insert_metric_name=False)
        checkpoint_last_epoch = ModelCheckpoint(filename="epoch={epoch:02d}-val_loss=val_loss{val/loss:.2f}",
                                                auto_insert_metric_name=False)
        model = UNet(1, 1, hparams["lr"], hparams["lossWeight"])
        trainer = pl.Trainer(accumulate_grad_batches=hparams["accumulateGradients"], gpus=1, max_epochs=hparams["epochs"], logger=logger,
                             precision=16, callbacks=[checkpoint_loss, checkpoint_last_epoch])
        logger.log_hyperparams(hparams)
        trainer.fit(model, DataLoader(train_data, batch_size=hparams["batchSize"], num_workers=8, pin_memory=True),
                    DataLoader(val_data, batch_size=hparams["batchSize"], num_workers=8, pin_memory=True))

        trainer.test(test_dataloaders=DataLoader(val_data, batch_size=1, shuffle=False))

        print("Done")
    except KeyboardInterrupt:
        print("Interrupted")
        if trainer and model:
            trainer.test(test_dataloaders=DataLoader(val_data, batch_size=1, shuffle=False))


