# Image segmentation project

The full project is explained [here](https://apiquet.com/2021/01/03/segmentation-model-implementation/)

In a few words, in this repository is implemented two architectures:

* VGG-16 + FCN-8 module. We use pre-trained weights from SSD300 implemented [here](https://github.com/Apiquet/Tracking_SSD_ReID)
Even if SSD300 is made for object dectection, its feature extractor can be reuse in another task with similar classes involved.
The article linked at the top of this readme explains the implementation and compares trainings with and without the transfer learning

The following illustration shows an overview of the architecture implemented:

![Transfer learning from SSD300](imgs/transfer_learning_from_ssd.png)

* The UNET architecture


![Person dog](imgs/person_dog_segmentation.gif)

![Person then dog](imgs/person_then_dog_segmentation.gif)

![Dog](imgs/dog_segmentation.gif)
