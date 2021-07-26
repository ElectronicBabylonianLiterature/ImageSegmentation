import pytorch_lightning as pl

from torch import optim, sigmoid
from torch.nn import BCEWithLogitsLoss
from torchmetrics.classification import Accuracy, Precision, Recall, F1

from .unet_parts import *


class UNet(pl.LightningModule):
    def __init__(self, n_channels, n_classes, lr, bilinear=True):
        super(UNet, self).__init__()
        self.counter = 1

        self.lr = lr
        self.loss = BCEWithLogitsLoss()

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def predict_step(self, batch, batch_idx, dataloader_idx = None):
        return (sigmoid(self(batch)) > 0.1).float()


    def configure_optimizers(self):
        optimizer = optim.RMSprop(self.parameters(), lr=self.lr, weight_decay=1e-8, momentum=0.9)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)
        return {"optimizer": optimizer, "scheduler": scheduler}

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log("loss", loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        val_loss = self.loss(y_hat, y)
        self.log("val_loss", val_loss)
        return val_loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log("test_loss", loss)
        prediction = self.predict_step(x, batch_idx)[0]

        tensorboard = self.logger.experiment
        tensorboard.add_image("mask/true", y[0], self.counter)
        tensorboard.add_image("mask/test", prediction, self.counter)
        self.counter = self.counter + 1
        return loss
