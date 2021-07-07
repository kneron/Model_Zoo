# Kneron AI Training Platform Model_Zoo

## Introduction

We provide a collection of classification and detection models pre-trained on the [ImageNet dataset](https://image-net.org/) and the [COCO dataset](http://cocodataset.org). In the table below, we summarized each such pre-trained model including: 
* a model name.
* model input size.
* model speed: we report frame per second (fps) evaluated on our 520 and 720 hardwares. 
* model size.
* model performance on the ImageNet validation set and COCO validation set.

## Pre-trained Classification Models
Model | Input Size | FPS on 520 | FPS on 720 | Model Size | Rank 1 Accuracy | Rank 5 Accuracy 
--- | --- |:---:|:---:|:---:|:---:|:---:
[mobilenetv2](https://github.com/kneron/Model_Zoo/tree/main/classification/MobileNetV2)| 224x224 | 58.9418 | 620.677 | 14M | 69.82% | 89.29%
[resnet18](https://github.com/kneron/Model_Zoo/tree/main/classification/ResNet18)| 224x224 | 20.4376 | 141.371 | 46.9M | 66.46% | 87.09%
[resnet50](https://github.com/kneron/Model_Zoo/tree/main/classification/ResNet50)| 224x224 | 6.32576 | 49.0828 | 102.9M | - | -
[FP_classifier](https://github.com/kneron/Model_Zoo/tree/main/classification/FP_classifier) | 56x32 | 323.471 | 3370.47 | 5.1M | 94.13% | -

[mobilenetv2](https://github.com/kneron/Model_Zoo/tree/main/classification/MobileNetV2),  [resnet18](https://github.com/kneron/Model_Zoo/tree/main/classification/ResNet18) and [resnet50](https://github.com/kneron/Model_Zoo/tree/main/classification/ResNet50) are models pre-trained on ImageNet classification dataset. [FP_classifier](https://github.com/kneron/Model_Zoo/tree/main/classification/FP_classifier) is a model pre-trained on our own dataset for classifying person and background images.

Resnet50 is currently under training for Kneron preprocessing.

## Pre-trained Detection Models
Backbone | Input Size |  FPS on 520 | FPS on 720  | Model Size | mAP
--- | --- |:---:|:---:|:---:|:---:
[YOLOv5s (no upsample)](https://github.com/kneron/Model_Zoo/tree/main/detection/yolov5/yolov5s-noupsample) | 640x640 | 4.91429 | - | 13.1M | 40.4%
[YOLOv5s (with upsample)](https://github.com/kneron/Model_Zoo/tree/main/detection/yolov5/yolov5s) | 640x640 | - | 24.4114 | 14.6M | 50.9%
[FCOS (darknet53s backbone)](https://github.com/kneron/Model_Zoo/tree/main/detection/yolov5/yolov5s) | 416x416 | 7.27369 | 48.8437 | 33.9M | 44.8%


