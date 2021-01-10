# Image segmentation project

The full project is explained [here](https://apiquet.com/2021/01/03/segmentation-model-implementation/)

In this repository is implemented two architectures:

* VGG-16 + FCN-8 module. We use pre-trained weights from SSD300 implemented [here](https://github.com/Apiquet/Tracking_SSD_ReID)
Although the SSD300 is designed for object detection, its feature extractor can be reused in another task involving similar classes.
The related article at the top of this readme explains the implementation and compares training with and without learning transfer.

The following illustration shows an overview of the architecture implemented:

![Transfer learning from SSD300](imgs/transfer_learning_from_ssd.png)

* The UNET architecture: [paper](https://arxiv.org/pdf/1505.04597.pdf)

![U-Net](imgs/unet_explained.png)


Some results with the first architecture:

![Person dog](imgs/person_dog_segmentation.gif)

![Person then dog](imgs/person_then_dog_segmentation.gif)

![Dog](imgs/dog_segmentation.gif)
