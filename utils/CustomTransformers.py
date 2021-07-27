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



class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, image):
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        #img = transform.resize(image, (new_h, new_w))

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        #landmarks = landmarks * [new_w / w, new_h / h]

        #return {'image': img, 'landmarks': landmarks}
        return None
