import torch
import torch.nn as nn
import PIL.Image as Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

path_img = '../res/Data/cat.png'

sampler = nn.Upsample(scale_factor=2, mode='bilinear')

img_pil = Image.open(path_img).convert('RGB')
plt.imshow(img_pil)
plt.show()

img_tensor = transforms.ToTensor()(img_pil).unsqueeze(0)
img_tensor = sampler(img_tensor).squeeze()
img_pil = transforms.ToPILImage()(img_tensor)
plt.imshow(img_pil)
plt.show()
