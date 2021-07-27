import numpy as np
import torch
from torchvision import transforms

from utils.elastic_transform import elastic_transform


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


