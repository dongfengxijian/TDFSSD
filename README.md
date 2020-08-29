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
