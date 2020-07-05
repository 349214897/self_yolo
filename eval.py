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
import xml.etree.ElementTree as ET
import os
import main_voc
import cv2
from backbone import trcnet

_dirname="/home/tan/e_work/datasets/VOC/VOC2007"
_split="val"
_is_2007= True
_show =False
_predictions = defaultdict(list)
_anno_file_template=os.path.join(_dirname, "Annotations", "{}.xml")
_image_set_path=os.path.join(_dirname, "ImageSets", "Main", _split + ".txt")
_save_path="/home/tan/e_work/result/%s.jpg"

def parse_rec(filename):
    """Parse a PASCAL VOC xml file."""
    with open(filename) as f:
        tree = ET.parse(f)
    objects = []
    for obj in tree.findall("object"):
        obj_struct = {}
        obj_struct["name"] = obj.find("name").text
        obj_struct["pose"] = obj.find("pose").text
        obj_struct["truncated"] = int(obj.find("truncated").text)
        obj_struct["difficult"] = int(obj.find("difficult").text)
        bbox = obj.find("bndbox")
        obj_struct["bbox"] = [
            int(bbox.find("xmin").text),
            int(bbox.find("ymin").text),
            int(bbox.find("xmax").text),
            int(bbox.find("ymax").text),
        ]
        objects.append(obj_struct)

    return objects

def voc_ap(rec, prec, use_07_metric=False):
    """Compute VOC AP given precision and recall. If use_07_metric is true, uses
    the VOC 07 11-point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.0
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.0], rec, [1.0]))
        mpre = np.concatenate(([0.0], prec, [0.0]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def voc_eval(detpath, annopath, imagesetfile, classname, ovthresh=0.5, use_07_metric=False):
    """rec, prec, ap = voc_eval(detpath,
                                annopath,
                                imagesetfile,
                                classname,
                                [ovthresh],
                                [use_07_metric])

    Top level function that does the PASCAL VOC evaluation.

    detpath: Path to detections
        detpath.format(classname) should produce the detection results file.
    annopath: Path to annotations
        annopath.format(imagename) should be the xml annotations file.
    imagesetfile: Text file containing the list of images, one image per line.
    classname: Category name (duh)
    [ovthresh]: Overlap threshold (default = 0.5)
    [use_07_metric]: Whether to use VOC07's 11 point AP computation
        (default False)
    """
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name

    # first load gt
    # read list of images
    with open(imagesetfile, "r") as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]

    # load annots
    recs = {}
    for imagename in imagenames:
        recs[imagename] = parse_rec(annopath.format(imagename))

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj["name"] == classname]
        bbox = np.array([x["bbox"] for x in R])
        difficult = np.array([x["difficult"] for x in R]).astype(np.bool)
        # difficult = np.array([False for x in R]).astype(np.bool)  # treat all "difficult" as GT
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {"bbox": bbox, "difficult": difficult, "det": det}

    # read dets
    detfile = detpath.format(classname)
    with open(detfile, "r") as f:
        lines = f.readlines()

    splitlines = [x.strip().split(" ") for x in lines]
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])
    BB = np.array([[float(z) for z in x[2:]] for x in splitlines]).reshape(-1, 4)

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        R = class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R["bbox"].astype(float)

        if BBGT.size > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1.0, 0.0)
            ih = np.maximum(iymax - iymin + 1.0, 0.0)
            inters = iw * ih

            # union
            uni = (
                (bb[2] - bb[0] + 1.0) * (bb[3] - bb[1] + 1.0)
                + (BBGT[:, 2] - BBGT[:, 0] + 1.0) * (BBGT[:, 3] - BBGT[:, 1] + 1.0)
                - inters
            )

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > ovthresh:
            if not R["difficult"][jmax]:
                if not R["det"][jmax]:
                    tp[d] = 1.0
                    R["det"][jmax] = 1
                else:
                    fp[d] = 1.0
        else:
            fp[d] = 1.0

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)

    return rec, prec, ap

def read_image_and_target(root_dir,split):
    import os
    _annopath = os.path.join(root_dir, "Annotations", "%s.xml")
    _imgpath = os.path.join(root_dir, "JPEGImages", "%s.jpg")
    _imgsetpath = os.path.join(root_dir, "ImageSets", "Main", "%s.txt")
    image_set = split

    with open(_imgsetpath % image_set) as f:
        ids = f.readlines()
    ids = [x.strip("\n") for x in ids]
    id_to_img_map = {k: v for k, v in enumerate(ids)}
    cls = voc.CLASSES
    class_to_ind = dict(zip(cls, range(len(cls))))
    categories = dict(zip(range(len(cls)), cls))


def process():
    net=trcnet.trcnet50()
    net=torch.load("/home/tan/e_work/project/self_yolo_anchorfree_iou_2/weights/iter_12000.pth")
    W,H=28,28
    IW,IH=448,448
    device = torch.device("cuda")
    net.to(device)
    transforms = transform.Transform()
    dataset= voc.PascalVOCDataset(_dirname, _split,transforms=transforms)
    g_target_ids=dataset.ids
    sample=torch.utils.data.RandomSampler(dataset)
    batch_size=1
    start_iter=0
    max_iter=100000
    # batch_sample=torch.utils.data.BatchSampler(sample,batch_size,False)
    # batch_sample=IterationBasedBatchSampler(batch_sample,max_iter,start_iter=start_iter)
    data_loader = torch.utils.data.DataLoader(dataset, collate_fn=BatchCollator(), batch_size=batch_size,
                                              num_workers=1)
    net.eval()  # start learning BN

    with torch.no_grad():
        for idx, (images, targets, data_ids) in enumerate(data_loader,0):
            output = net(torch.stack(images).cuda())
            bsize, _, h, w = output.size()
            output = output.permute(0, 2, 3, 1).contiguous().view(bsize, -1, 1, 26)
            bbox_pred = torch.sigmoid(output[:, :, :, 0:4])
            iou_pred = F.sigmoid(output[:, :, :, 4:5])

            score_pred = output[:, :, :, 5:].contiguous()
            prob_pred = F.softmax(score_pred.view(-1, score_pred.size()[-1])).view_as(score_pred)

            bbox_pred_np = bbox_pred.data.cpu().numpy()
            iou_pred_np = iou_pred.data.cpu().numpy()

            _boxes, _ious, _classes, _box_mask, _iou_mask, _class_mask = main_voc.build_target(bbox_pred_np, iou_pred_np,
                                                                                      targets)
            _boxes = main_voc.np_to_variable(_boxes)
            _ious = main_voc.np_to_variable(_ious)

            bbox_np = no_anchor_to_bbox(
                np.ascontiguousarray(bbox_pred_np, dtype=np.float),
                H, W)
            # bbox_np = (hw, num_anchors, (x1, y1, x2, y2))   range: 0 ~ 1

            bbox_np[:, :, :, 0::2] *= float(IW)  # rescale x
            bbox_np[:, :, :, 1::2] *= float(IH)  # rescale y

            scores = iou_pred.cpu().numpy()
            cats = prob_pred.cpu().numpy()
            for b in range(bsize):
                bbox_bs=bbox_np[b]
                cat_bs=cats[b]
                score_bs=scores[b].reshape(scores[b].shape[0])
                image_id=g_target_ids[data_ids[b]]

                if(_show):
                    image=main_voc.show_image(images[b],bbox_pred[b],iou_pred[b],_boxes[b],_ious[b],prob_pred[b],_classes[b],True)
                    cv2.imshow("image",image)
                    cv2.waitKey(0)
                else:
                    image = main_voc.show_image(images[b], bbox_pred[b], iou_pred[b], _boxes[b], _ious[b], prob_pred[b],
                                                _classes[b], True)
                    cv2.imwrite(_save_path%(image_id),image)

                mask=score_bs>0.9
                bbox_bs=bbox_bs[mask]
                score_bs=score_bs[mask]
                cat_bs=cat_bs[mask]
                bbox_bs=bbox_bs[:,0,:]
                keep=main_voc.nms(bbox_bs,score_bs,0.5)
                bbox_bs=bbox_bs[keep]
                score_bs=score_bs[keep]
                cat_bs=cat_bs[keep]
                cls=cat_bs.argmax(axis=2)
                for e in range(len(bbox_bs)):
                    s="%s %.3f %.1f %.1f %.1f %.1f"%(image_id,score_bs[e],bbox_bs[e,0],bbox_bs[e,1],bbox_bs[e,2],bbox_bs[e,3])
                    # print(s," ",cls[e,0])
                    _predictions[cls[e,0]].append(
                        s
                    )

    import pickle
    buffer = pickle.dumps(_predictions)
    if len(buffer) > 1024 ** 3:
        print(
            "Rank {} trying to all-gather {:.2f} GB of data on device {}"
        )
    predictions = _predictions.copy()

    import tempfile
    import os
    classes=dataset.CLASSES
    with tempfile.TemporaryDirectory(prefix="pascal_voc_eval_") as dirname:
        res_file_template = os.path.join(dirname, "{}.txt")

        aps = defaultdict(list)  # iou -> ap per class
        for cls_id, cls_name in enumerate(classes):
            lines = predictions.get(cls_id, [""])

            with open(res_file_template.format(cls_name), "w") as f:
                f.write("\n".join(lines))

            for thresh in range(50, 100, 5):
                rec, prec, ap = voc_eval(
                    res_file_template,
                    _anno_file_template,
                    _image_set_path,
                    cls_name,
                    ovthresh=thresh / 100.0,
                    use_07_metric=_is_2007,
                )
                aps[thresh].append(ap * 100)

    ret = OrderedDict()
    mAP = {iou: np.mean(x) for iou, x in aps.items()}
    ret["bbox"] = {"AP": np.mean(list(mAP.values())), "AP50": mAP[50], "AP75": mAP[75]}
    return ret

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
    ret=process()
    print(ret)