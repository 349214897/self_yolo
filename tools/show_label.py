import os
import cv2
import sys
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

rootdir="/home/tan/e_work/datasets/VOC/VOC2007"
_anno_file_template = os.path.join(rootdir, "Annotations", "{}.xml")
_image_set_path = os.path.join(rootdir, "ImageSets", "Main", "val_small" + ".txt")
save_path="/home/tan/e_work/show_label"

with open(_image_set_path) as f:
    for e in f.readlines():
        image_id=e.strip()
        image_name=image_id+".jpg"
        image_path=os.path.join(rootdir,"JPEGImages",image_name)
        annofile=os.path.join(rootdir,"Annotations",image_id+".xml")
        print(e.strip())

        image=cv2.imread(image_path)
        # cv2.imshow("image",image)
        # cv2.waitKey(0)
        target = ET.parse(annofile).getroot()
        for obj in target.iter("object"):
            bb = obj.find("bndbox")
            # Make pixel indexes 0-based
            # Refer to "https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/datasets/pascal_voc.py#L208-L211"
            box = [
                bb.find("xmin").text,
                bb.find("ymin").text,
                bb.find("xmax").text,
                bb.find("ymax").text,
            ]
            bndbox = tuple(
                map(lambda x: x - 1, list(map(int, box)))
            )
            cv2.rectangle(image,(bndbox[0],bndbox[1]),(bndbox[2],bndbox[3]),(0,255,0),2)
        cv2.imwrite(os.path.join(save_path,image_name),image)