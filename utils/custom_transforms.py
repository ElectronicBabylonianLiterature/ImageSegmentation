import warnings
from collections import Sequence

import numpy as np
import torch
from torch.autograd.grad_mode import F
from torchvision import transforms
from torchvision.transforms.functional import _interpolation_modes_from_int, InterpolationMode

from utils.elastic_transform import elastic_transform

class BinarizationTransform(object):
    def __init__(self, threshold):
        self.threshold = threshold

    def __call__(self, image):
        result = image > self.threshold
        return result.float()

    def __repr__(self):
        return self.__class__.__name__ + '()'
class ElasticTransform(object):
    def __call__(self, pic):
        x = pic.shape[3]
        random_state = np.random.RandomState(x)
        image = transforms.ToTensor()(elastic_transform(pic[0][0].numpy(), random_state=random_state))
        random_state1 = np.random.RandomState(x)
        target = transforms.ToTensor()(elastic_transform(pic[1][0].numpy(), random_state=random_state1))
        return torch.stack([image,target])

    def __repr__(self):
        return self.__class__.__name__ + '()'


class DynamicResize(torch.nn.Module):
    def __init__(self, size, interpolation=InterpolationMode.BILINEAR, max_size=None, antialias=None):
        super().__init__()
        self.size = tuple([self.resizeDict(i[0], i[1]) for i in size])
        self.max_size = max_size

        # Backward compatibility with integer value
        if isinstance(interpolation, int):
            warnings.warn(
                "Argument interpolation should be of type InterpolationMode instead of int. "
                "Please, use InterpolationMode enum."
            )
            interpolation = _interpolation_modes_from_int(interpolation)

        self.interpolation = interpolation
        self.antialias = antialias

    def resizeDict(self, max_value, rescale):
        return {"maxValue": max_value, "rescale": rescale}

    def resizeImage(self, shape, resizing_params):
        x, y = shape
        max_value = max(x, y)
        min_value = min(x, y)
        for i in range(0, len(resizing_params) - 1):
            scaling = resizing_params[i + 1]["rescale"]
            if resizing_params[i]["maxValue"] < max_value < resizing_params[i + 1]["maxValue"]:
                return min_value / scaling
            elif max_value > resizing_params[len(resizing_params) - 1]["maxValue"]:
                scaling = resizing_params[len(resizing_params) - 1]["rescale"]
                return min_value / scaling
        return min_value

    def forward(self, img):
        new_size_smaller_dimension = self.resizeImage(img.shape[-2:], self.size)
        return transforms.Resize(int(new_size_smaller_dimension), self.interpolation, self.max_size, self.antialias)(img)