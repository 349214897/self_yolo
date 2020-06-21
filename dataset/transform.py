import torch
from torchvision.transforms import functional as F
import cv2
import numpy as np
class Transform(object):
    def __call__(self,image,target):
        image = F.resize(image, (448,448))
        target = target.resize((448,448))

        # imageshow=np.transpose(image,(0,1,2))
        # print(imageshow.shape)
        # tmp_box=target.convert("xyxy")
        # for e in tmp_box.bbox:
        #     print(e)
        #     cv2.rectangle(imageshow,(e[0],e[1]),(e[2],e[3]),(255,0,0))
        # cv2.imshow("img",imageshow)
        # cv2.waitKey(0)

        image=F.to_tensor(image)
        print("!!!!!!!!!!!!!!ok")
        return image,target
