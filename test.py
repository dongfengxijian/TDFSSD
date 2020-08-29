#Written by Haodong Pan Email:860782934@qq.com
from __future__ import print_function

import argparse

import cv2
import numpy as np
import os
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import matplotlib.pyplot as plt

from data import BaseTransform, VOC_300, VOC_512, COCO_300, COCO_512
from data import VOC_CLASSES as labelmap
#from data import COCO_CLASSES as labelmap
from layers.functions import Detect, PriorBox
from utils.timer import Timer

parser = argparse.ArgumentParser(description='Top-Down Feature Fusion Single Shot MultiBox Detector')

parser.add_argument('-v', '--version', default='TDFSSD_vgg',
                    help='Sorry only TDFSSD_vgg is supported currently!')
parser.add_argument('-s', '--size', default='300',
                    help='300 or 512 input size.')
parser.add_argument('-d', '--dataset', default='VOC',
                    help='VOC or COCO version')
parser.add_argument('-m', '--trained_model',
                    default='/home/phd/PycharmProjects/TDFSSD/weights/TDFSSD_vgg_300/TDFSSD_vgg_VOC_epoches_280.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='eval/', type=str,
                    help='Dir to save results')
parser.add_argument('--cuda', default=False, type=bool,
                    help='Use cuda to train model')
parser.add_argument('--retest', default=False, type=bool,
                    help='test cache results')
args = parser.parse_args()

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

if args.dataset == 'VOC':
    cfg = (VOC_300, VOC_512)[args.size == '512']
else:
    cfg = (COCO_300, COCO_512)[args.size == '512']

if args.version == 'TDFSSD_vgg':
    from models.TDFSSD_vgg import build_net
else:
    print('Unkown version!')

priorbox = PriorBox(cfg)
priors = Variable(priorbox.forward())


def py_cpu_nms(dets, thresh):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


class ObjectDetector:
    def __init__(self, net, detection, transform, num_classes=21, cuda=False, max_per_image=300, thresh=0.5):
        self.net = net
        self.detection = detection
        self.transform = transform
        self.max_per_image = 300
        self.num_classes = num_classes
        self.max_per_image = max_per_image
        self.cuda = cuda
        self.thresh = thresh

    def predict(self, img):
        scale = torch.Tensor([img.shape[1], img.shape[0],
                              img.shape[1], img.shape[0]]).cpu().numpy()
        _t = {'im_detect': Timer(), 'misc': Timer()}
        assert img.shape[2] == 3
        x = Variable(self.transform(img).unsqueeze(0))
        if self.cuda:
            x = x.cuda()
        _t['im_detect'].tic()
        out = net(x, test=True)  # forward pass
        boxes, scores = self.detection.forward(out, priors)
        detect_time = _t['im_detect'].toc()
        boxes = boxes[0]
        scores = scores[0]

        boxes = boxes.cpu().numpy()
        scores = scores.cpu().numpy()
        # scale each detection back up to the image
        boxes *= scale
        _t['misc'].tic()
        all_boxes = [[] for _ in range(num_classes)]

        for j in range(1, num_classes):
            inds = np.where(scores[:, j] > self.thresh)[0]
            if len(inds) == 0:
                all_boxes[j] = np.zeros([0, 5], dtype=np.float32)
                continue
            c_bboxes = boxes[inds]
            c_scores = scores[inds, j]
            print(scores[:, j])
            c_dets = np.hstack((c_bboxes, c_scores[:, np.newaxis])).astype(
                np.float32, copy=False)

            keep = py_cpu_nms(c_dets, 0.45)
            keep = keep[:50]
            c_dets = c_dets[keep, :]
            all_boxes[j] = c_dets
        if self.max_per_image > 0:
            image_scores = np.hstack([all_boxes[j][:, -1] for j in range(1, num_classes)])
            if len(image_scores) > self.max_per_image:
                image_thresh = np.sort(image_scores)[-self.max_per_image]
                for j in range(1, num_classes):
                    keep = np.where(all_boxes[j][:, -1] >= image_thresh)[0]
                    all_boxes[j] = all_boxes[j][keep, :]

        nms_time = _t['misc'].toc()
        print('net time: ', detect_time)
        print('post time: ', nms_time)
        return all_boxes


COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
FONT = cv2.FONT_HERSHEY_SIMPLEX

if __name__ == '__main__':
    # load net
    img_dim = (300, 512)[args.size == '512']
    num_classes = (21, 81)[args.dataset == 'COCO']
    net = build_net(img_dim, num_classes)  # initialize detector
    state_dict = torch.load(args.trained_model, map_location=lambda storage, loc: storage)
    # create new OrderedDict that does not contain `module.`

    from collections import OrderedDict

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        head = k[:7]
        if head == 'module.':
            name = k[7:]  # remove `module.`
        else:
            name = k
        new_state_dict[name] = v
    net.load_state_dict(new_state_dict)
    net.eval()
    print('Finished loading model!')

    # load data
    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True
    # evaluation
    top_k = (300, 200)[args.dataset == 'COCO']
    detector = Detect(num_classes, 0, cfg)
    rgb_means = ((104, 117, 123), (103.94, 116.78, 123.68))[args.version == 'RFB_mobile']
    rgb_std = (1, 1, 1)
    transform = BaseTransform(net.size, rgb_means, rgb_std, (2, 0, 1))
    object_detector = ObjectDetector(net, detector, transform, num_classes=num_classes, max_per_image=top_k, thresh=0.01)
    colors = plt.cm.hsv(np.linspace(0, 1, num_classes)).tolist()

    directory_name = 'examples/images/ComDPM'
    for filename in os.listdir(r"./"+directory_name):
        img = cv2.imread(directory_name + "/" + filename)
        image = cv2.resize(img, (640, 480))
        b, g, r = cv2.split(image)
        img2 = cv2.merge([r, g, b])
        plt.rcParams['figure.dpi'] = 600
        plt.imshow(img2)  # expect true color
        currentAxis = plt.gca()
        detect_bboxes = object_detector.predict(image)
        for class_id, class_collection in enumerate(detect_bboxes):
            if len(class_collection) > 0:
               for i in range(class_collection.shape[0]):
                   if class_collection[i, -1] > 0.6:
                      pt = class_collection[i]
                      display_txt = '%s: %.2f' % (labelmap[class_id], pt[-1])
                      coords = (int(pt[0]), int(pt[1])), int(pt[2])-int(pt[0])+1, int(pt[3])-int(pt[1])+1
                      color = colors[class_id]
                      currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
                      currentAxis.text(int(pt[0]), int(pt[1]), display_txt, bbox={'facecolor': color, 'alpha': 0.5})
        plt.axis('off')
        plt.savefig('examples/result/ComDPM/'+'/'+filename)
        plt.show()
        plt.close()


