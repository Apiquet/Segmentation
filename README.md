# Image segmentation project

The project is explained [here](https://apiquet.com/2021/01/03/segmentation-model-implementation/)

In this repository is implemented three architectures: VGG-16 + FCN-8 module, VGG-16 + FCN-4 module and U-Net.

The two models with VGG-16 use pre-trained weights from SSD300 model implemented [here](https://github.com/Apiquet/Tracking_SSD_ReID).
Although the SSD300 is designed for object detection, its feature extractor can be reused in another task involving similar classes.
The related article (link at the top of this readme) explains the implementation and compares training with and without transfer learning.
It also describes how to parse raw data to train segmentation models.

* FCN-8 architecture and some visualizations:

![FCN8](imgs/fcn8.png)

![Person dog](imgs/fcn8_example1.gif)

![Person then dog](imgs/fcn8_example2.gif)

![Dog](imgs/fcn8_example3.gif)

* FCN-4 architecture and some visualizations:

![FCN4](imgs/fcn4.png)

![Person dog](imgs/fcn4_example1.gif)

![Person then dog](imgs/fcn4_example2.gif)

![Dog](imgs/fcn4_example3.gif)

* U-NET architecture and a visualization: [paper](https://arxiv.org/pdf/1505.04597.pdf)

![U-Net](imgs/unet.png)

![Dog-UNet](imgs/unet_example1.gif)

## Usage

* Training: the notebooks UNET/FCN4/8_training.ipynb show how to train a UNET / VGG-16 + FCN-4 / VGG-16 + FCN8 models.
* Testing: the notebook infer_on_videos.ipynb shows how to infer the segmentation model VGG-16 + FCN-4 on a single image or on a video.

The script under utils/ folder allows to create the visualizations
