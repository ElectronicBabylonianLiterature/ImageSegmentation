import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms

SEG_LABELS_LIST1 = [
                {"id": 0, "name": "background",  "greyscale_value": 0},
                {"id": 1,  "name": "line",   "greyscale_value": 255}
]



class SegmentationDataset(Dataset):
    def __init__(self, image_paths_file, resize,  binarization_threshold, transform=None, normalize=None):
        self.root_dir_name = os.path.dirname(image_paths_file)
        self.transform = transform
        self.resize = resize
        self.binarization_threshold = binarization_threshold
        self.normalize= normalize

        with open(image_paths_file) as f:
            self.image_names = f.read().splitlines()

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, key):
        if isinstance(key, slice):
            return [self[ii] for ii in range(*key.indices(len(self)))]
        elif isinstance(key, int):
            if key < 0:
                key += len(self)
            if key < 0 or key >= len(self):
                raise IndexError("The index (%d) is out of range." % key)
            return self.get_item_from_index(key)
        else:
            raise TypeError("Invalid argument type.")

    def resizeImage(self, image, resizing_params):
        x, y = image.size
        for i in range(0, len(resizing_params) - 1):
            scaling = resizing_params[i+1]["rescale"]
            if resizing_params[i]["maxValue"] < max(x, y) < resizing_params[i+1]["maxValue"]:
                return image.resize((int(x / scaling), int(y / scaling)))
            elif max(x, y) > resizing_params[len(resizing_params) -1]["maxValue"]:
                scaling = resizing_params[len(resizing_params) - 1]["rescale"]
                return image.resize((int(x / scaling), int(y / scaling)))
        return image



    def get_item_from_index(self, index):
        img_id = self.image_names[index]

        img = Image.open(os.path.join(self.root_dir_name, "images", f"{img_id}.png"))
        img = self.resizeImage(img, self.resize)
        toTensor = transforms.ToTensor()
        img = toTensor(img)

        target = Image.open(os.path.join(self.root_dir_name, "targets", f"{img_id}_GT0.png")).convert("L")
        target = self.resizeImage(target, self.resize).point(lambda p: 0 if p < self.binarization_threshold * 255 else 1)
        target = np.array(target, dtype=np.int)
        target = np.array([target])
        target_labels = torch.from_numpy(target.copy()).float()
        result = torch.stack([img, target_labels])
        if self.transform:
            result = self.transform(result)
        return self.normalize(result[0]) if self.normalize else result[0], result[1], img_id


