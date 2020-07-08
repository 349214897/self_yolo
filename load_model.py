from sfs_dw1 import Yolo
from sfs_dw1 import SfsVps
from backbone import trcnet
from backbone import trcnet14
from torchvision import models
import torch

def load_model(cfg,mode):
    arch=cfg.MODEL.ARCH
    if(mode=="train"):
        if(arch=="trcnet"):
            net = trcnet.trcnet50()
            # load pretrain resnet50
            model_dict = net.state_dict()
            res_net = models.resnet50(pretrained=True)
            pretrained_dict = res_net.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            net.load_state_dict(model_dict)

            # net=torch.load("/home/tan/e_work/project/self_yolo_anchorfree_iou/weights/iter_200.pth")
            device = torch.device("cuda")
            net.to(device)
        if(arch=="trcnet14"):
            net = trcnet14.trcnet50()
            # load pretrain resnet50
            model_dict = net.state_dict()
            res_net = models.resnet50(pretrained=True)
            pretrained_dict = res_net.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            net.load_state_dict(model_dict)

            # net=torch.load("/home/tan/e_work/project/self_yolo_anchorfree_iou/weights/iter_200.pth")
            device = torch.device("cuda")
            net.to(device)
    if(mode=="test"):
        net = trcnet.trcnet50()
        net = torch.load(cfg.EVAL.MODEL_PATH)

        device = torch.device("cuda")
        net.to(device)
    if(mode=="predict"):
        net = SfsVps(cfg=None)
        net = torch.load(cfg.EVAL.MODEL_PATH)
    return net