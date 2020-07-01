from collections import OrderedDict, defaultdict
from sfs_dw1 import SfsVps
from main_voc import no_anchor_to_bbox
from main_voc import IterationBasedBatchSampler
from main_voc import BatchCollator
from dataset import voc
from dataset import transform
import torch
import torch.nn.functional as F
import numpy as np

_predictions = defaultdict(list)


def process():
    net=SfsVps(cfg=None)
    net=torch.load("/home/tan/e_work/project/self_yolo_anchorfree_iou_2/weights/iter_8400.pth")
    W,H=14,14
    IW,IH=448,448
    device = torch.device("cuda")
    net.to(device)
    transforms = transform.Transform()
    dataset= voc.PascalVOCDataset("/home/tan/e_work/datasets/VOC/VOC2012", "trainval",transforms=transforms)
    sample=torch.utils.data.RandomSampler(dataset)
    batch_size=8
    start_iter=0
    max_iter=100000
    batch_sample=torch.utils.data.BatchSampler(sample,batch_size,False)
    batch_sample=IterationBasedBatchSampler(batch_sample,max_iter,start_iter=start_iter)
    data_loader = torch.utils.data.DataLoader(dataset, collate_fn=BatchCollator(), batch_sampler=batch_sample,
                                              num_workers=4)
    net.eval()  # start learning BN

    with torch.no_grad():
        for idx, (images, targets, _) in enumerate(data_loader, 0):
            output = net(torch.stack(images).cuda())
            bsize, _, h, w = output.size()
            output = output.permute(0, 2, 3, 1).contiguous().view(bsize, -1, 1, 26)
            bbox_pred = torch.sigmoid(output[:, :, :, 0:4])
            iou_pred = F.sigmoid(output[:, :, :, 4:5])

            score_pred = output[:, :, :, 5:].contiguous()
            prob_pred = F.softmax(score_pred.view(-1, score_pred.size()[-1])).view_as(score_pred)

            bbox_pred_np = bbox_pred.data.cpu().numpy()
            iou_pred_np = iou_pred.data.cpu().numpy()

            bbox_np = no_anchor_to_bbox(
                np.ascontiguousarray(bbox_pred_np, dtype=np.float),
                H, W)
            # bbox_np = (hw, num_anchors, (x1, y1, x2, y2))   range: 0 ~ 1

            bbox_np[:, :, :, 0::2] *= float(IW)  # rescale x
            bbox_np[:, :, :, 1::2] *= float(IH)  # rescale y
            score = iou_pred
            cat = prob_pred
            for b in range(bsize):
                bboxes=bbox_np[b,:,:,0:4].view(-1,4)


    # for input, output in zip(inputs, outputs):
    #     image_id = input["image_id"]
    #     boxes = instances.pred_boxes.tensor.numpy()
    #     scores = instances.scores.tolist()
    #     classes = instances.pred_classes.tolist()
    #     for box, score, cls in zip(boxes, scores, classes):
    #         xmin, ymin, xmax, ymax = box
    #         # The inverse of data loading logic in `datasets/pascal_voc.py`
    #         xmin += 1
    #         ymin += 1
    #         _predictions[cls].append(
    #             f"{image_id} {score:.3f} {xmin:.1f} {ymin:.1f} {xmax:.1f} {ymax:.1f}"
    #         )

if __name__=="__main__":
    process()