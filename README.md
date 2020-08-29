# TDFSSD: Top-Down Feature Fusion Single Shot MultiBox Detector
By Haodong Pan, Jue Jiang, Guangfeng Chen

## Introduction
This is the code for our accepted manuscript in Signal Processing: Image Communication. In brief, this is a SSD based approach for object detection across different scales.  
If you use this code, please cite our paper.

## Prerequisities
Python3.6, PyTorch0.4.1, and NVIDIA GPUs

## Installation
* Install PyTorch-0.4.1 depending on your environment refering to https://pytorch.org/.  
* Clone this repository. This repository is mainly based on[RFBNet](https://github.com/ruinmessi/RFBNet), [ssd.pytorch](https://github.com/amdegroot/ssd.pytorch), [Chainer-ssd](https://github.com/Hakuyume/chainer-ssd) and [PytorchSSD](https://github.com/lzx1413/PytorchSSD), a huge thank to them. 
* Compile the nms and coco tools:  
```Shell
./make.sh
```
Check your GPU architecture support in utils/build.py, line 131. Default is:
``` 
'nvcc': ['-arch=sm_61',
```
Then download the dataset by following the [instructions](#download-voc2007-trainval--test) below and install opencv. 
```Shell
conda install opencv
```
## Datasets
For convenience, we provide simple VOC and COCO dataset loader.

### VOC Dataset
##### Download VOC2007 trainval & test
```Shell
# specify a directory for dataset to be downloaded into, else default is ~/data/
sh data/scripts/VOC2007.sh # <directory>
```

##### Download VOC2012 trainval
```Shell
# specify a directory for dataset to be downloaded into, else default is ~/data/
sh data/scripts/VOC2012.sh # <directory>
```

### COCO Dataset
Install the MS COCO dataset at /path/to/coco from [official website](http://mscoco.org/), default is ~/data/COCO. It should have this basic structure
```Shell
$COCO/  
$COCO/annotations/
$COCO/cache/
$COCO/cocoapi/
$COCO/images/
$COCO/test2017/
$COCO/train2017/
$COCO/val2017/
```  

## Training  
- Download the fc-reduced [VGG-16](https://arxiv.org/abs/1409.1556) PyTorch base network weights at our [BaiduYun Driver](https://pan.baidu.com/s/1XtKJGWU0nyyNUC0gDICMRw) PW:1234
- The modified [resnet-101](https://arxiv.org/pdf/1512.03385.pdf) PyTorch base network weights file is available at our [BaiduYun Driver](https://pan.baidu.com/s/1BQnwMrrmtcZeuBsApgXoQw) PW:1234
- Place the weights files in 'TDFSSD/weights'
- To train TDFSSD with the following command:
```shell
python train_test.py -d VOC -s 300 -we 6
```  
- Note:  
  * -d: datasets, VOC or COCO
  * -s: image size, 300 or 512
  * -we: warm epoch 
  * 'resnet_trian_test.py' is used for training with the resnet as backbone
  * The detail options can be found in the 'train_test.py' and 'resnet_trian_test.py'
 
## Evaluation
To evaluate a trained network with 'test.py' or 'eval.py' 

## Models
* 07+12 [TDFSSD300](https://pan.baidu.com/s/1jLZz-46iPFNJDLcf09pYIA) PW:1234
* COCO [TDFSSD300](https://pan.baidu.com/s/12WpB-Lu7L5MaXKky3lwtng) PW:1234
