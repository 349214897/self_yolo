import torch
from torchvision.transforms import functional as F
import cv2
import numpy as np
class Transform(object):
    def __call__(self,image,target):
        image = F.resize(image, (448,448))
        target = target.resize((448,448))
        image=F.to_tensor(image)
        return image,target
