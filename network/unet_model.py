import pytorch_lightning as pl
from torch import optim, sigmoid
from torch.nn import BCEWithLogitsLoss
from torchmetrics.functional import dice_score, accuracy, precision, recall, f1, iou

from .unet_parts import *


class UNet(pl.LightningModule):
    def __init__(self, n_channels, n_classes, lr, bilinear=True):
        super(UNet, self).__init__()

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
        x, _, _ = batch
        return sigmoid(self(x))


    def configure_optimizers(self):
        optimizer = optim.RMSprop(self.parameters(), lr=self.lr, weight_decay=1e-8, momentum=0.9)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        return {"optimizer": optimizer}

    def training_step(self, batch, batch_idx):
        x, y, _ = batch

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        y_hat = self.outc(x)

        loss = self.loss(y_hat, y)
        self.log("Loss/train", loss, on_step=False, on_epoch=True)

        """
        #takes around 1GB VRAM
        acc_score = self.calculate_metric(batch, accuracy, batch_idx)
        recall_score = self.calculate_metric(batch, recall, batch_idx)
        precision_score = self.calculate_metric(batch, precision, batch_idx)
        f1_score = self.calculate_metric(batch, f1, batch_idx)
        self.log_dict({"Accuracy/train": acc_score, "Recall/train": recall_score, "Precision/train": precision_score, "F1/train": f1_score}, on_step=False, on_epoch=True)
        """
        return loss

    def calculate_metric(self, batch, metric, batch_idx=None):
        x, y, _ = batch
        prediction = self.predict_step((x,None, None), batch_idx)
        return metric(prediction, y.int())

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self(x)
        val_loss = self.loss(y_hat, y)
        self.log("Loss/val", val_loss, on_step=False, on_epoch=True)
        return val_loss

    def test_step(self, batch, batch_idx):
        x, y, img_id = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log("Loss/test", loss)

        acc_score = self.calculate_metric(batch, accuracy, batch_idx)
        recall_score = self.calculate_metric(batch, recall, batch_idx)
        precision_score = self.calculate_metric(batch, precision, batch_idx)
        f1_score = self.calculate_metric(batch, f1, batch_idx)
        iou_score = self.calculate_metric(batch, iou, batch_idx)

        self.log_dict({"Accuracy/test": acc_score, "IoU/test": iou_score, "F1/test": f1_score, "Recall/test": recall_score, "Precision/test": precision_score}, on_step=False, on_epoch=True)

        predicions = self.predict_step((x, None, None), batch_idx)
        self.logger.experiment.add_image(f"test/mask-{img_id}/true", y[0],)
        self.logger.experiment.add_image(f"test/mask-{img_id}/prediction", predicions[0])
        self.logger.experiment.add_pr_curve("precision-recall curve", y[0], predicions[0])

        return loss
