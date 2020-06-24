import os
from PIL import Image
from torchvision.transforms import functional as F
from sfs_dw1 import SfsVps
import main_voc
import cv2
import torch
import numpy as np

#load weights
root_path="/home/tan/e_work/datasets/VOC/VOC2012/JPEGImages"
lines=os.listdir(root_path)
net=SfsVps(cfg=None)
net = torch.load("/media/tan/B4FE-5315/self_yolo/weights/iter_31400.pth")
# net.load_state_dict(state_dict)
net.eval()

device = torch.device("cuda")
net.to(device)
with torch.no_grad():
    for line in lines:
        image_path=os.path.join(root_path,line)
        image=Image.open(image_path ).convert("RGB")

        image = F.resize(image, (448, 448))
        image = F.to_tensor(image)
        image_expand = image.view((1,3,448,448))
        output=net(image_expand.cuda())

        bsize, _, h, w = output.size()
        output = output.permute(0, 2, 3, 1).contiguous().view(bsize, -1, 1, 26)
        import torch.nn.functional as F1
        # tx, ty, tw, th, to -> sig(tx), sig(ty), exp(tw), exp(th), sig(to)
        xy_pred = F1.sigmoid(output[:, :, :, 0:2])
        # wh_pred = torch.exp(output[:, :, :, 2:4])
        wh_pred = torch.sigmoid(output[:, :, :, 2:4])
        bbox = torch.cat([xy_pred, wh_pred], 3)
        score = F1.sigmoid(output[:, :, :, 4:5])

        import torchvision
        transforms=torchvision.transforms
        image = transforms.ToPILImage()(image).convert('RGB')
        image=np.transpose(image,(0,1,2))

        bbox=bbox[0]
        score=score[0]

        pred_bbox_np=bbox.cpu().detach().numpy()
        pred_score_np=score.cpu().detach().numpy()

        anchors=np.array([[14,14]])
        H,W=14,14
        O_H,O_W =448,448

        pred_bbox_np = np.expand_dims(pred_bbox_np, 0)
        pred_score_mask=pred_score_np[:,0,0]>0.7

        pred_bbox=main_voc.yolo_to_bbox(pred_bbox_np,H,W,anchors)
        pred_bbox[:,:, :, 0::2] *= float(O_W)  # rescale x
        pred_bbox[:,:, :, 1::2] *= float(O_H)  # rescale y
        pred_bbox=pred_bbox[0]

        pred_bbox=pred_bbox[pred_score_mask,0,:]
        keep=main_voc.nms(pred_bbox,pred_score_np[pred_score_mask,0,0],0.3)

        for idx in keep:
            pt1=(int(pred_bbox[idx,0]),int(pred_bbox[idx,1]))
            pt2=(int(pred_bbox[idx,2]),int(pred_bbox[idx,3]))
            cv2.rectangle(image,pt1,pt2,(0,255,0))

        cv2.imshow("image_res",image)
        cv2.waitKey(0)



