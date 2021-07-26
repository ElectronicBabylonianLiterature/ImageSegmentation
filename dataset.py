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
    def __init__(self, image_paths_file, resize, transform=None):
        self.root_dir_name = os.path.dirname(image_paths_file)
        self.transform = transform
        self.resize = resize

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


    def get_item_from_index(self, index):
        img_id = self.image_names[index]

        img = Image.open(os.path.join(self.root_dir_name, "images", f"{img_id}.png")).resize(self.resize)
        toTensor = transforms.ToTensor()
        img = toTensor(img)

        target = Image.open(os.path.join(self.root_dir_name, "targets", f"{img_id}_GT0.png")).resize(self.resize).convert("L").point(lambda p: 0 if p < 50 else 1)
        target = np.array(target, dtype=np.int)
        target = np.array([target])
        target_labels = torch.from_numpy(target.copy()).float()

        if self.transform:
            img = self.transform(img)
            target_labels = self.transform(target_labels)

        return img, target_labels
