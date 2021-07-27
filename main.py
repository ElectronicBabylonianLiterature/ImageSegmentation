import os

import torch
from pytorch_lightning.loggers import TensorBoardLogger
from torchvision import transforms

from CustomTransformers import ElasticTransform
from dataset import SegmentationDataset
from visualization import binarization_threshold_values

root = os.path.dirname(os.path.abspath(__file__))
data_root = os.path.join(root, 'segmentation_data')

logger = TensorBoardLogger("logs", name="u_net", log_graph=True)

def resizeDict(maxValue, rescale):
    return {"maxValue": maxValue, "rescale": rescale}

def calculate_mean_and_std(training_images_path, resize):
    train_data = SegmentationDataset(image_paths_file=training_images_path, resize=resize, binarization_threshold=0.1)

    size = len(train_data)
    meanTotal = 0
    stdTotal = 0
    for (img, target, img_id) in train_data:
        mean, std = torch.std_mean(img)
        meanTotal = meanTotal + mean
        stdTotal = stdTotal + std

    return meanTotal/size, stdTotal/size



resize = (resizeDict(1000, 1), resizeDict(2000, 2), resizeDict(4000, 3),  resizeDict(float("inf"), 4))

mean, std = calculate_mean_and_std(training_images_path=f"{data_root}/train.txt", resize=resize)


randomElastic = transforms.RandomApply([ElasticTransform()], 0.25)
randomAffine = transforms.RandomApply([transforms.RandomAffine(degrees=(-15, 15), scale=(0.7, 1.5))], 0.5)
transform = transforms.Compose([randomAffine, randomElastic, transforms.Normalize(mean, std)])




hparams = {
    "batchSize": 1,
    "epochs": 1,
    "lr": 0.001,
    "resizeImages": resize,
    "transformations": transform,
    "binarization_threshold": 0.1,
    "mean": mean,
    "std": std,
}

train_data = SegmentationDataset(image_paths_file=f"{data_root}/train.txt",resize=hparams["resizeImages"], binarization_threshold=hparams["binarization_threshold"], transform=hparams["transformations"])
test_data = SegmentationDataset(image_paths_file=f"{data_root}/test.txt", resize=hparams["resizeImages"], binarization_threshold=hparams["binarization_threshold"], transform=hparams["transformations"])


hparams["trainSize"] = len(train_data)
hparams["testSize"] = len(test_data)

print("Train size: %i" % len(train_data))
print("Test size: %i" % len(test_data))
print("Img size: ", train_data[0][0].size())
print("Segmentation size: ", train_data[0][1].size())


#binarization_threshold_values(f"{data_root}/train.txt",hparams["resizeImages"], "binarization_tests", transform)

for counter, index in enumerate(np.random.randint(0, len(train_data) - 1, 10)):
    i = int(index)
    logger.experiment.add_image(f"train_samples/sample-{counter}/training", train_data[i][0])
    logger.experiment.add_image(f"train_samples/sample-{counter}/truth", train_data[i][1])


model = UNet(n_channels=1, n_classes=1, lr=hparams["lr"])
logger.log_graph(model)

trainer = pl.Trainer(gpus=1, max_epochs=hparams["epochs"], logger=logger, log_every_n_steps=1)
logger.log_hyperparams(hparams)
trainer.fit(model, DataLoader(train_data, batch_size=hparams["batchSize"], num_workers=4))

trainer.test(model, DataLoader(test_data, batch_size=1, shuffle=False))
torch.save(model.state_dict(), f"./models/u-net.pt")

print("Done")

