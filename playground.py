import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from torchvision import transforms

img1 = Image.open("/home/yunus/PycharmProjects/aru-net/segmentation_data/images/K.1156.png").convert("L")
np_img1 = np.array(img1)
print(np.amax(np_img1))
x = transforms.ToTensor()(np_img1)
print(np.amax(np.array(x)))

plt.figure()
plt.imshow(np_img1, cmap="gray")

img2 = Image.open("/home/yunus/PycharmProjects/aru-net/segmentation_data/targets/K.1156_GT0.png").convert("1")
np_img2 = np.array(img2)
print(np.amax(np_img2))
x = transforms.ToTensor()(np_img2)
print(np.amax(np.array(x)))
plt.figure()
plt.imshow(np_img2, cmap="gray")


plt.show()

to_tensor = transforms.ToTensor()
img = to_tensor(img1)
print("Done")