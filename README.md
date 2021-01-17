# Image segmentation project

The full project is explained [here](https://apiquet.com/2021/01/03/segmentation-model-implementation/)

In this repository is implemented two architectures:

* VGG-16 + FCN-8 module. We use pre-trained weights from SSD300 implemented [here](https://github.com/Apiquet/Tracking_SSD_ReID)
Although the SSD300 is designed for object detection, its feature extractor can be reused in another task involving similar classes.
The related article at the top of this readme explains the implementation and compares training with and without learning transfer.

* FCN-8 architectures and some results:

![FCN8](imgs/fcn8.png)

![Person dog](imgs/fcn8_example1.gif)

![Person then dog](imgs/fcn8_example2.gif)

![Dog](imgs/fcn8_example3.gif)

* FCN-4 architectures and some results:

![FCN4](imgs/fcn4.png)

![Person dog](imgs/fcn4_example1.gif)

![Person then dog](imgs/fcn4_example2.gif)

![Dog](imgs/fcn4_example3.gif)

* U-NET architecture and some results: [paper](https://arxiv.org/pdf/1505.04597.pdf)

![U-Net](imgs/unet.png)

![Dog-UNet](imgs/unet_example1.gif)

