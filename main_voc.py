import torch
import os
import torch.optim as optim
from sfs_dw1 import Yolo
from sfs_dw1 import SfsVps
import cv2
from collections import defaultdict
import numpy as np
import torch.nn as nn
import torch.utils.data
from dataset import voc
from dataset import transform
from torch.autograd import Variable
from torch.utils.data.sampler import BatchSampler

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

class BatchCollator(object):
    """
    From a list of samples from the dataset,
    returns the batched images and targets.
    This should be passed to the DataLoader
    """

    def __init__(self, size_divisible=0):
        self.size_divisible = size_divisible

    def __call__(self, batch):
        transposed_batch = list(zip(*batch))
        images = transposed_batch[0]
        targets = transposed_batch[1]
        img_ids = transposed_batch[2]
        return images, targets, img_ids

class IterationBasedBatchSampler(BatchSampler):
    """
    Wraps a BatchSampler, resampling from it until
    a specified number of iterations have been sampled
    """

    def __init__(self, batch_sampler, num_iterations, start_iter=0):
        self.batch_sampler = batch_sampler
        self.num_iterations = num_iterations
        self.start_iter = start_iter

    def __iter__(self):
        iteration = self.start_iter
        while iteration <= self.num_iterations:
            # if the underlying sampler has a set_epoch method, like
            # DistributedSampler, used for making each process see
            # a different split of the dataset, then set it
            if hasattr(self.batch_sampler.sampler, "set_epoch"):
                self.batch_sampler.sampler.set_epoch(iteration)
            for batch in self.batch_sampler:
                iteration += 1
                if iteration > self.num_iterations:
                    break
                yield batch

    def __len__(self):
        return self.num_iterations

def yolo_to_bbox(bbox_pred,H,W,anchors):
    bsize=bbox_pred.shape[0]
    num_anchors = anchors.shape[0]

    bbox_out = np.zeros((bsize,H*W,num_anchors,4),dtype=float)
    for b in range(bsize):
        for row in range(H):
            for col in range(W):
                ind = row * W + col
                for a in range(num_anchors):
                    cx = (bbox_pred[b, ind, a, 0] + col) / W
                    cy = (bbox_pred[b, ind, a, 1] + row) / H
                    bw = bbox_pred[b, ind, a, 2] * anchors[a][0] / W * 0.5
                    bh = bbox_pred[b, ind, a, 3] * anchors[a][1] / H * 0.5

                    bbox_out[b, ind, a, 0] = cx - bw
                    bbox_out[b, ind, a, 1] = cy - bh
                    bbox_out[b, ind, a, 2] = cx + bw
                    bbox_out[b, ind, a, 3] = cy + bh
    return bbox_out

def build_target(bbox_pred_np,iou_pred_np,targets):
    bsize = bbox_pred_np.shape[0]

    H,W=14,14
    inp_size=[448,448]
    out_size=[14,14]
    anchors=np.array([[3.6,3.6]])
    num_classes=21
    coord_scale=1.0
    class_scale=1.0
    object_scale=1.0
    noobject_scale = 0.3
    iou_thresh = 0.5


    hw,num_anchors,_=bbox_pred_np[bsize-1].shape
    _classes = np.zeros([bsize,hw,num_anchors,num_classes],dtype=np.float)
    _class_mask= np.zeros([bsize,hw,num_anchors,num_classes],dtype=np.float)

    _ious = np.zeros([bsize,hw, num_anchors, 1], dtype=np.float)
    _iou_mask = np.zeros([bsize,hw,num_anchors, 1], dtype=np.float)

    _boxes = np.zeros([bsize,hw, num_anchors, 4], dtype=np.float)
    _boxes[:,:, :, 0:2] = 0.5
    _boxes[:,:, :, 2:4] = 1.0
    _box_mask = np.zeros([bsize,hw, num_anchors, 1], dtype=np.float) + 0.01


    # scale pred_bbox

    # bbox_pred_np = np.expand_dims(bbox_pred_np, 0)
    bbox_np = yolo_to_bbox(
        np.ascontiguousarray(bbox_pred_np, dtype=np.float),
        H, W,anchors)
    # bbox_np = (hw, num_anchors, (x1, y1, x2, y2))   range: 0 ~ 1

    bbox_np[:,:, :, 0::2] *= float(inp_size[0])  # rescale x
    bbox_np[:,:, :, 1::2] *= float(inp_size[1])  # rescale y

    # gt_boxes=np.stack(tuple(boxlist.bbox for boxlist in targets))
    # gt_classes=np.stack(tuple(boxlist.get_field("labels") for boxlist in targets))
    # gt_boxes_b = np.asarray(gt_boxes[b], dtype=np.float)
    # gt_boxes = np.asarray(gt_boxes, dtype=np.float)

    # for each cell, compare predicted_bbox and gt_bbox
    for b in range(bsize):
        bbox_np_b = np.reshape(bbox_np[b], [-1, 4])
        gt_boxes_b=targets[b].bbox
        gt_classes_b=targets[b].get_field("labels")
        ious = bbox_ious(
            np.ascontiguousarray(bbox_np_b, dtype=np.float),
            np.ascontiguousarray(gt_boxes_b, dtype=np.float)
        )
        best_ious = np.max(ious, axis=1).reshape(_iou_mask[b].shape)
        iou_penalty = 0 - iou_pred_np[b,best_ious < iou_thresh]
        _iou_mask[b,best_ious <= iou_thresh] = noobject_scale * iou_penalty

        # locate the cell of each gt_boxe
        cell_w = float(inp_size[0]) / W
        cell_h = float(inp_size[1]) / H
        cx = (gt_boxes_b[:, 0] + gt_boxes_b[:, 2]) * 0.5 / cell_w
        cy = (gt_boxes_b[:, 1] + gt_boxes_b[:, 3]) * 0.5 / cell_h
        cell_inds = np.floor(cy) * W + np.floor(cx)
        cell_inds = cell_inds.int()

        target_boxes = np.empty(gt_boxes_b.shape, dtype=np.float)
        target_boxes[:, 0] = cx - np.floor(cx)  # cx
        target_boxes[:, 1] = cy - np.floor(cy)  # cy
        target_boxes[:, 2] = \
            (gt_boxes_b[:, 2] - gt_boxes_b[:, 0]) / inp_size[0] * out_size[0]  # tw
        target_boxes[:, 3] = \
            (gt_boxes_b[:, 3] - gt_boxes_b[:, 1]) / inp_size[1] * out_size[1]  # th

        ious_reshaped = np.reshape(ious, [hw, num_anchors, len(cell_inds)])
        for i, cell_ind in enumerate(cell_inds):
            if cell_ind >= hw or cell_ind < 0:
                print('cell inds size {}'.format(len(cell_inds)))
                print('cell over {} hw {}'.format(cell_ind, hw))
                continue
            # a = anchor_inds[i]
            a=0
            # 0 ~ 1, should be close to 1
            iou_pred_cell_anchor = iou_pred_np[b,cell_ind, a, :]
            _iou_mask[b,cell_ind, a, :] = object_scale * (1 - iou_pred_cell_anchor)  # noqa
            # _ious[cell_ind, a, :] = anchor_ious[a, i]
            # _ious[b,cell_ind, a, :] = ious_reshaped[cell_ind, a, i]
            _ious[b, cell_ind, a, :] = 1.0

            _box_mask[b,cell_ind, a, :] = coord_scale
            target_boxes[i, 2:4] /= anchors[a]
            _boxes[b,cell_ind, a, :] = target_boxes[i]

            _class_mask[b,cell_ind, a, :] = class_scale
            _classes[b,cell_ind, a, gt_classes_b[i]] = 1.
    return _boxes, _ious, _classes, _box_mask, _iou_mask, _class_mask

def bbox_ious(boxes,query_boxes):
    """
    For each query box compute the IOU covered by boxes
    ----------
    Parameters
    ----------
    boxes: (N, 4) ndarray of float
    query_boxes: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of intersec between boxes and query_boxes
    """
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    intersec = np.zeros((N, K),dtype=float)

    for k in range(K):
        qbox_area = (
            (query_boxes[k, 2] - query_boxes[k, 0] + 1) *
            (query_boxes[k, 3] - query_boxes[k, 1] + 1)
        )
        for n in range(N):
            iw = (
                min(boxes[n, 2], query_boxes[k, 2]) -
                max(boxes[n, 0], query_boxes[k, 0]) + 1
            )
            if iw > 0:
                ih = (
                    min(boxes[n, 3], query_boxes[k, 3]) -
                    max(boxes[n, 1], query_boxes[k, 1]) + 1
                )
                if ih > 0:
                    box_area = (
                        (boxes[n, 2] - boxes[n, 0] + 1) *
                        (boxes[n, 3] - boxes[n, 1] + 1)
                    )
                    inter_area = iw * ih
                    intersec[n, k] = inter_area / (qbox_area + box_area - inter_area)
    return intersec

def np_to_variable(x, is_cuda=True, dtype=torch.FloatTensor, volatile=False):
    v = Variable(torch.from_numpy(x).type(dtype), volatile=volatile)
    if is_cuda:
        v = v.cuda()
    return v

def nms(dets, scores, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    ndets = dets.shape[0]
    suppressed = np.zeros((ndets), dtype=np.int)

    keep = []
    for _i in range(ndets):
        i = order[_i]
        if suppressed[i] == 1:
            continue
        keep.append(i)
        ix1 = x1[i]
        iy1 = y1[i]
        ix2 = x2[i]
        iy2 = y2[i]
        iarea = areas[i]
        for _j in range(_i + 1, ndets):
            j = order[_j]
            if suppressed[j] == 1:
                continue
            xx1 = max(ix1, x1[j])
            yy1 = max(iy1, y1[j])
            xx2 = min(ix2, x2[j])
            yy2 = min(iy2, y2[j])
            w = max(0.0, xx2 - xx1 + 1)
            h = max(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (iarea + areas[j] - inter)
            if ovr >= thresh:
                suppressed[j] = 1

    return keep

def show_image(image,bbox,score,target_box,target_score):
    #PILImage ->to_tensor->torch->ToPilImage->cvimage
    import torchvision
    transforms=torchvision.transforms
    imageshow = transforms.ToPILImage()(image).convert('RGB')
    imageshow=np.transpose(imageshow,(0,1,2))

    pred_bbox_np=bbox.cpu().detach().numpy()
    pred_score_np=score.cpu().detach().numpy()
    target_bbox_np=target_box.cpu().numpy()
    target_score_np=target_score.cpu().numpy()

    anchors=np.array([[3.6,3.6]])
    H,W=14,14
    O_H,O_W =448,448

    pred_bbox_np = np.expand_dims(pred_bbox_np, 0)
    target_bbox_np = np.expand_dims(target_bbox_np,0)

    pred_score_mask=pred_score_np[:,0,0]>0.3
    pred_bbox=yolo_to_bbox(pred_bbox_np,H,W,anchors)
    pred_bbox[:,:, :, 0::2] *= float(O_W)  # rescale x
    pred_bbox[:,:, :, 1::2] *= float(O_H)  # rescale y
    pred_bbox=pred_bbox[0]

    target_score_mask=target_score_np[:,0,0]>0.8
    target_bbox=yolo_to_bbox(target_bbox_np,H,W,anchors)
    target_bbox[:,:, :, 0::2] *= float(O_W)  # rescale x
    target_bbox[:,:, :, 1::2] *= float(O_H)  # rescale y
    target_bbox=target_bbox[0]

    pred_bbox=pred_bbox[pred_score_mask,0,:]
    keep=nms(pred_bbox,pred_score_np[pred_score_mask,0,0],0.7)
    for idx in keep:
        pt1=(int(pred_bbox[idx,0]),int(pred_bbox[idx,1]))
        pt2=(int(pred_bbox[idx,2]),int(pred_bbox[idx,3]))
        cv2.rectangle(imageshow,pt1,pt2,(0,255,0))


    target_bbox=target_bbox[target_score_mask,0,:]
    keep=nms(target_bbox,target_score_np[target_score_mask,0,0],0.7)
    for idx in keep:
        pt1=(int(target_bbox[idx,0]),int(target_bbox[idx,1]))
        pt2=(int(target_bbox[idx,2]),int(target_bbox[idx,3]))
        cv2.rectangle(imageshow,pt1,pt2,(255,0,0))

    # for e in target.bbox:
    #     cv2.rectangle(imageshow,(e[0],e[1]),(e[2],e[3]),(255,0,0))
    return imageshow


def train():
    # parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    # parser.add_argument(
    #     "--",
    #     default="",
    #     metavar="FILE",
    #     help="path to config file",
    #     type=str,
    # )


    # net=Yolo(cfg=None)
    net=SfsVps(cfg=None)
    device = torch.device("cuda")
    net.to(device)
    # create your optimizer
    optimizer = optim.SGD(net.parameters(), lr=0.01)
    # in your training loop:
    optimizer.zero_grad()  # zero the gradient buffers
    from tensorboardX import SummaryWriter
    writer=SummaryWriter('log')

    transforms = transform.Transform()
    dataset= voc.PascalVOCDataset("/home/tan/e_work/datasets/VOC/VOC2012", "trainval",transforms=transforms)

    sample=torch.utils.data.RandomSampler(dataset)
    batch_size=128
    start_iter=0
    max_iter=100000
    W,H=14,14
    batch_sample=torch.utils.data.BatchSampler(sample,batch_size,False)
    batch_sample=IterationBasedBatchSampler(batch_sample,max_iter,start_iter=start_iter)

    train_loss = 0
    bbox_loss, iou_loss, cls_loss = 0., 0., 0.

    # dataloader
    data_loader=torch.utils.data.DataLoader(dataset,collate_fn=BatchCollator(),batch_sampler=batch_sample,num_workers=4)
    net.train()  #start learning BN

    if(os.path.exists("weights")==False):
        os.mkdir("weights")
    for idx,(images,targets,_) in enumerate(data_loader,0):
        output=net(torch.stack(images).cuda())
        bsize, _, h, w = output.size()
        output=output.permute(0,2,3,1).contiguous().view(bsize,-1,1,26)

        # tx, ty, tw, th, to -> sig(tx), sig(ty), exp(tw), exp(th), sig(to)
        xy_pred = F.sigmoid(output[:, :, :, 0:2])
        wh_pred = torch.exp(output[:, :, :, 2:4])
        bbox_pred = torch.cat([xy_pred, wh_pred], 3)
        iou_pred = F.sigmoid(output[:, :, :, 4:5])

        score_pred = output[:, :, :, 5:].contiguous()
        prob_pred = F.softmax(score_pred.view(-1, score_pred.size()[-1])).view_as(score_pred)

        bbox_pred_np = bbox_pred.data.cpu().numpy()
        iou_pred_np = iou_pred.data.cpu().numpy()
        _boxes, _ious, _classes, _box_mask, _iou_mask, _class_mask=build_target(bbox_pred_np,iou_pred_np,targets)
        _boxes = np_to_variable(_boxes)
        _ious = np_to_variable(_ious)
        _classes = np_to_variable(_classes)
        box_mask = np_to_variable(_box_mask,
                                            dtype=torch.FloatTensor)
        iou_mask = np_to_variable(_iou_mask,
                                            dtype=torch.FloatTensor)
        class_mask = np_to_variable(_class_mask,
                                              dtype=torch.FloatTensor)

        num_boxes = sum((boxes.bbox.shape[0] for boxes in targets))

        # _boxes[:, :, :, 2:4] = torch.log(_boxes[:, :, :, 2:4])
        box_mask = box_mask.expand_as(_boxes)

        bbox_loss = nn.MSELoss(size_average=False)(bbox_pred * box_mask, _boxes * box_mask) / num_boxes  # noqa
        iou_loss = nn.MSELoss(size_average=False)(iou_pred * iou_mask, _ious * iou_mask) / num_boxes  # noqa

        class_mask = class_mask.expand_as(prob_pred)
        cls_loss = nn.MSELoss(size_average=False)(prob_pred * class_mask, _classes * class_mask) / num_boxes  # noqa
        loss=bbox_loss+iou_loss+cls_loss

        print("iter: %d, loss: %f, lcls: %f, lobj: %f ,lbox: %f"%(idx,loss,cls_loss,iou_loss,bbox_loss))
        if(idx%1==0):
            # write for tensorboard
            writer.add_scalar("loss", loss, idx)
            writer.add_scalar("lcls", cls_loss, idx)
            writer.add_scalar("lobj", iou_loss, idx)
            writer.add_scalar("lbox", bbox_loss, idx)

            imageshow=show_image(images[0],bbox_pred[0],iou_pred[0],_boxes[0],_ious[0])
            writer.add_image("image", torch.from_numpy(imageshow).permute(2, 0, 1), idx)
            writer.add_image("score", output[0,:,0,4].view(1, W, H), idx)
            writer.add_image("target_score", _ious[0].view(1, W, H), idx)
        if(idx%200==0):
            torch.save(net, "weights/iter_%d.pth" % (idx))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return

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