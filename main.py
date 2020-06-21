import torch
import argparse
import os
import torch.optim as optim
from sfs_dw1 import Yolo
import json
import cv2
from collections import defaultdict
import numpy as np
import torch.nn as nn
from PIL import Image


import torch.nn.functional as F

def cal_iou(box1,box2):
    # box2 = box2.t()


    b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
    b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
    b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
    b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter = (min(b1_x2, b2_x2) - max(b1_x1, b2_x1)) * \
            (min(b1_y2, b2_y2) - max(b1_y1, b2_y1))

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
    union = (w1 * h1 + 1e-16) + w2 * h2 - inter

    iou = inter / union  # iou
    return iou

def train():
    # parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    # parser.add_argument(
    #     "--",
    #     default="",
    #     metavar="FILE",
    #     help="path to config file",
    #     type=str,
    # )


    net=Yolo(cfg=None)
    device = torch.device("cuda")
    net.to(device)
    # create your optimizer
    optimizer = optim.SGD(net.parameters(), lr=0.01)
    # in your training loop:
    optimizer.zero_grad()  # zero the gradient buffers
    import tensorboardX
    from tensorboardX import SummaryWriter
    writer=SummaryWriter('log')

    annotation_name="instances_val2017.json"
    sub_data_path="val2017"
    # read data
    coco_data_path="/home/tan/e_work/datasets/coco"
    annotation_file=os.path.join(coco_data_path,"annotations",annotation_name)
    dataset = json.load(open(annotation_file, 'r'))
    categories=dataset["categories"]
    images=dataset["images"]
    annotations=dataset["annotations"]
    #1.
    ids=[]
    ims={}
    id2name={}
    id2trainid={}
    img2annos=defaultdict(list)
    for e in images:
        ids.append(e["id"])
    for e in images:
        ims[e["id"]]=e
    for e in annotations:
        img2annos[e["image_id"]].append(e)
    for train_id,e in enumerate(categories):
        id2name[e["id"]]=e["name"]
        id2trainid[e["id"]]=train_id

    epoch=0
    while(epoch<20):
        epoch=epoch+1
        for iter,idx in enumerate(ids):
            image_full_name=os.path.join(coco_data_path,sub_data_path,ims[idx]["file_name"])
            image=cv2.imread(image_full_name)
            image_copy=image[...]
            annos=img2annos[idx]
            clses=[e["category_id"] for e in annos]
            bboxes=[e["bbox"] for e in annos]
            indexes=[e["category_id"] for e in annos]


            # show label
            # for box,id in zip(bboxes,indexes):
            #     x1=int(box[0])
            #     y1=int(box[1])
            #     x2=int(box[0]+box[2])
            #     y2=int(box[1]+box[3])
            #     cv2.rectangle(image,(x1,y1),(x2,y2),(255,0,0))
            #
            #     cls_name=id2name[id]
            #     cv2.putText(image,cls_name,(x1,y1),1,1,(0,255,0))
            # cv2.imshow("label",image)




            scal1 = float(448)/image.shape[1]/4
            scal2 = float(448)/image.shape[0]/4
            image=cv2.resize(image,(448,448))
            image_copy = cv2.resize(image_copy, (448, 448))
            #ready label
            S=112
            C=80
            B=1
            target=np.zeros((B*5+C,S,S))
            bbox=np.asanyarray(bboxes)
            if(bbox.shape[0]==0):
                print("box is None")
                continue
            bbox[:,0]*=scal1
            bbox[:,2]*=scal1
            bbox[:,1]*=scal2
            bbox[:,3]*=scal2

            # for box in bbox:
            #     x1=int(box[0])*4
            #     y1=int(box[1])*4
            #     x2=int(box[0]+box[2])*4
            #     y2=int(box[1]+box[3])*4
            #     cv2.rectangle(image_copy,(x1,y1),(x2,y2),(255,0,0))
            #
            # cv2.imshow("label_scale",image_copy)

            bbox[:,0]=bbox[:,0]+bbox[:,2]/2
            bbox[:,1]=bbox[:,1]+bbox[:,3]/2

            # bbox[:,2]=(bbox[:,2]+bbox[:,0])*2
            # bbox[:,3]=(bbox[:,3]+bbox[:,1])*2

            # show label





            clses=[id2trainid[e] for e in clses]
            clses=np.asanyarray(clses)
            offset_score=0
            offset_cls=5
            offset_x=1
            offset_y=2
            offset_w=3
            offset_h=4
            maskx=np.logical_and(bbox[:,0].astype(int)>=0 , bbox[:,0].astype(int)<112)
            masky=np.logical_and(bbox[:,1].astype(int)>=0 , bbox[:,1].astype(int)<112)
            mask=np.logical_and(maskx,masky)
            bbox = bbox[mask]
            clses =clses[mask]


            target[0,bbox[:,1].astype(int),bbox[:,0].astype(int)]=1

            target[offset_cls+clses.astype(int),bbox[:,1].astype(int),bbox[:,0].astype(int)]=1
            target[offset_x, bbox[:,1].astype(int),bbox[:,0].astype(int)] = bbox[:,0]-bbox[:,0].astype(int)
            target[offset_y, bbox[:,1].astype(int),bbox[:,0].astype(int)] = bbox[:,1] - bbox[:,1].astype(int)
            target[offset_w, bbox[:,1].astype(int),bbox[:,0].astype(int)] = bbox[:,2] - bbox[:,2].astype(int)
            target[offset_h, bbox[:,1].astype(int),bbox[:,0].astype(int)] = bbox[:,3] - bbox[:,3].astype(int)

            #show target
            # cv2.imshow("score",target[0,])
            # cv2.imshow("img",image)
            # cv2.waitKey(0)





            input=torch.Tensor(image).permute(2,0,1).view(-1,3,448,448).cuda()
            output=net(input)


            # cal loss
            # Define criteria
            BCEcls = nn.BCEWithLogitsLoss()
            BCEobj = nn.BCEWithLogitsLoss()
            ft = torch.cuda.FloatTensor if output[0].is_cuda else torch.Tensor
            lcls, lbox, lobj = ft([0]), ft([0]), ft([0])
            mask=target[0][...]
            for idx in range(bbox.shape[0]):
                idxi,idxj=bbox[idx,1].astype(int),bbox[idx,0].astype(int)
                # cal iou
                p_box_x=F.sigmoid(output[0,offset_x,idxi,idxj])+idxj
                p_box_y=F.sigmoid(output[0,offset_y,idxi,idxj])+idxi
                p_box_w=F.sigmoid(output[0,offset_w,idxi,idxj])
                p_box_h = F.sigmoid(output[0,offset_h, idxi, idxj])
                box1=(p_box_x,p_box_y,p_box_w,p_box_h)
                box2=bbox[idx]
                iou=cal_iou(box1,box2)

                # draw image
                # image=np.zeros((112,112,3),dtype='uint8')
                # b1_x1, b1_x2 = int(box1[0] - box1[2] / 2), int(box1[0] + box1[2] / 2)
                # b1_y1, b1_y2 = int(box1[1] - box1[3] / 2), int(box1[1] + box1[3] / 2)
                # b2_x1, b2_x2 = int(box2[0] - box2[2] / 2), int(box2[0] + box2[2] / 2)
                # b2_y1, b2_y2 = int(box2[1] - box2[3] / 2), int(box2[1] + box2[3] / 2)
                # cv2.rectangle(image,(b1_x1,b1_y1),(b1_x2,b1_y2),(255,0,0),1)
                # cv2.rectangle(image, (b2_x1, b2_y1), (b2_x2, b2_y2), (0, 255, 0), 1)
                # cv2.imshow("image",image)
                # cv2.waitKey(0)
                # print("iou: %f , liou: %f"%(iou,(1.0 - iou).sum()))
                lbox+=(1.0 - iou).sum()

                lcls+=BCEcls(output[0,offset_cls+clses[idx],idxi,idxj].float(),torch.tensor(1).float().cuda())

                score_src=output[0, 0, idxi, idxj].float()
                score_sig=F.sigmoid(score_src)
                sub_lobj=BCEobj(output[0,0,idxi,idxj].float(),torch.tensor(1).float().cuda())
                print("score_src: %f score_sig: %f sub_lobj: %f"%(score_src,score_sig,sub_lobj))
                lobj+=sub_lobj

                # for tensorboard write
                box1=(p_box_x,p_box_y,p_box_w,p_box_h)
                box2=bbox[idx]
                b1_x1, b1_x2 = int(box1[0] - box1[2] / 2), int(box1[0] + box1[2] / 2)
                b1_y1, b1_y2 = int(box1[1] - box1[3] / 2), int(box1[1] + box1[3] / 2)
                b2_x1, b2_x2 = int(box2[0] - box2[2] / 2), int(box2[0] + box2[2] / 2)
                b2_y1, b2_y2 = int(box2[1] - box2[3] / 2), int(box2[1] + box2[3] / 2)
                cv2.rectangle(image_copy,(b1_x1*4,b1_y1*4),(b1_x2*4,b1_y2*4),(255,0,0))
                cv2.rectangle(image_copy, (b2_x1 * 4, b2_y1 * 4), (b2_x2 * 4, b2_y2 * 4), (0, 255, 0))

            # cv2.imshow("label_scale",image_copy)
            cv2.waitKey(0)
            lcls/=bbox.shape[0]
            lobj/=bbox.shape[0]
            lbox/=bbox.shape[0]

            loss=lcls+lobj+lbox
            # score_loss_fuc=nn.MSELoss()
            # loss=score_loss_fuc(torch.from_numpy(target[0]).float(),output[0].float())
            print("loss: %f lcls: %f lobj: %f lbox: %f"%(loss,lcls,lobj,lbox))

            writer.add_scalar("loss",loss,iter)
            writer.add_scalar("lcls",lcls,iter)
            writer.add_scalar("lobj", lobj,iter)
            writer.add_scalar("lbox", lbox,iter)

            # image_pil = Image.fromarray(cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB))
            writer.add_image("image",torch.from_numpy(image_copy).permute(2,0,1),iter)
            writer.add_image("score", output[0,0].view(1,112,112), iter)
                # writer.add_custom_scalars({'sum loss': loss, 'lcls': lcls, 'lobj': lobj,'lbox':lbox})
            if(iter%10000==0):
                torch.save(net,"weights/iter_%d_%d.pth"%(epoch,iter))




            optimizer.zero_grad()
            loss.backward()
            optimizer.step()  # Does the update
            # scheduler.step()
    writer.close()


if __name__=="__main__":
    train()

    #1.view,transpose,reshape,permute
    # a=torch.randn(2,3,4)
    # print a
    # print a.transpose(1,2)

    #2.numpy index
    # a=np.arange(24).reshape(4,6)
    # t=torch.from_numpy(a)
    # print(t)
    # print(t.t())
    # target_boxes=t[:,2:6]
    # gxy=target_boxes[:,:2]
    #
    # gwh=target_boxes[:,2:]
    # b,target_labels=t[:,:2].long().t()
    # gx,gy=gxy.t()
    # gw,gh=gwh.t()
    # # gi,gj=gxy.long().t()
    # ByteTensor=torch.ByteTensor
    # FloatTensor=torch.FloatTensor
    # obj_mask = ByteTensor(19, 1, 12, 12).fill_(0)
    # b,cls=t[:,:2].t()
    # print(b)
    # print(cls)
    # print(obj_mask)
    # obj_mask[b,:,3,3]=1
    # print(obj_mask)

    ##logical
    a=np.ones((5,3))*True
    b=np.ones((5,3))*False
    a[a[:,:2]==True]=False
    print(a)





    print("ok")