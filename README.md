# TDFSSD: Top-Down Feature Fusion Single Shot MultiBox Detector
By Haodong Pan, Jue Jiang, Guangfeng Chen
## Introduction
This is the code for our accepted manuscript in Signal Processing: Image Communication. In brief, this is a SSD based approach for object detection across different scales.  
If you use this code, please cite our paper.
## Prerequisities
Python3.6, PyTorch0.4.1, and NVIDIA GPUs
## Installation
* Install PyTorch-0.4.1 according to your environment refering to https://pytorch.org/.  
* Clone this repository.  
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
- First download the fc-reduced [VGG-16](https://arxiv.org/abs/1409.1556) PyTorch base network weights at our [BaiduYun Driver]() 
