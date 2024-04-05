# ERA-Session-10

## Target
* [x] Write a custom to an external site. ResNet architecture for CIFAR10 that has the following architecture:
  * PrepLayer - Conv 3x3 s1, p1) >> BN >> RELU [64k]
  * Layer1 -
    * X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [128k]
    * R1 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [128k] 
    * Add(X, R1)
  * Layer 2 -
    * Conv 3x3 [256k]
    * MaxPooling2D
    * BN
    * ReLU
  * Layer 3 -
    * X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [512k]
    * R2 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [512k]
    * Add(X, R2)
  * MaxPooling with Kernel Size 4
  * FC Layer 
  * SoftMax
* [x] Uses One Cycle Policy such that:
  * Total Epochs = 24
  * Max at Epoch = 5
  * LRMIN = FIND
  * LRMAX = FIND
  * NO Annihilation
* [x] Uses this transform -RandomCrop 32, 32 (after padding of 4) >> FlipLR >> Followed by CutOut(8, 8)
* [x] Batch size = 512
* [x] Use ADAM and CrossEntropyLoss
* [x] Target Accuracy: 90%

## CIFAR10 Sample Images
![image](https://github.com/ShubhamVerma16/ERA-Session-10/assets/46774613/32df26ac-5e2b-496f-9380-8d2f6875eeff)

## Model Summary
```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 64, 32, 32]           1,728
       BatchNorm2d-2           [-1, 64, 32, 32]             128
              ReLU-3           [-1, 64, 32, 32]               0
            Conv2d-4          [-1, 128, 32, 32]          73,728
         MaxPool2d-5          [-1, 128, 16, 16]               0
       BatchNorm2d-6          [-1, 128, 16, 16]             256
              ReLU-7          [-1, 128, 16, 16]               0
            Conv2d-8          [-1, 128, 16, 16]         147,456
       BatchNorm2d-9          [-1, 128, 16, 16]             256
             ReLU-10          [-1, 128, 16, 16]               0
           Conv2d-11          [-1, 128, 16, 16]         147,456
      BatchNorm2d-12          [-1, 128, 16, 16]             256
             ReLU-13          [-1, 128, 16, 16]               0
         ResBlock-14          [-1, 128, 16, 16]               0
           Conv2d-15          [-1, 256, 16, 16]         294,912
        MaxPool2d-16            [-1, 256, 8, 8]               0
      BatchNorm2d-17            [-1, 256, 8, 8]             512
             ReLU-18            [-1, 256, 8, 8]               0
           Conv2d-19            [-1, 512, 8, 8]       1,180,160
        MaxPool2d-20            [-1, 512, 4, 4]               0
      BatchNorm2d-21            [-1, 512, 4, 4]           1,024
             ReLU-22            [-1, 512, 4, 4]               0
           Conv2d-23            [-1, 512, 4, 4]       2,359,296
      BatchNorm2d-24            [-1, 512, 4, 4]           1,024
             ReLU-25            [-1, 512, 4, 4]               0
           Conv2d-26            [-1, 512, 4, 4]       2,359,296
      BatchNorm2d-27            [-1, 512, 4, 4]           1,024
             ReLU-28            [-1, 512, 4, 4]               0
         ResBlock-29            [-1, 512, 4, 4]               0
        MaxPool2d-30            [-1, 512, 1, 1]               0
           Linear-31                   [-1, 10]           5,120
          Softmax-32                   [-1, 10]               0
================================================================
Total params: 6,573,632
Trainable params: 6,573,632
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 6.75
Params size (MB): 25.08
Estimated Total Size (MB): 31.84
----------------------------------------------------------------
```

## LR Finder

## LR Plot

## Learning Curve
