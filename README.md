# COMP3314-Group10 README

## Introduction
This repository is for the code sharing of COMP3314 group project.
The codes are implemented based on the reference https://www.kaggle.com/code/mubtasimahasan/cifar10-with-cnn-and-googlenet, which has been released under the Apache 2.0 license. 

## The paper details
Title: An Analysis Of Convolutional Neural Networks For Image Classification
Authors: Neha Sharma,Vibhor Jain, Anju Mishra
Venue: Procedia Computer Science, Volume 132, 2018, Pages 377-384, ISSN 1877-0509, https://doi.org/10.1016/j.procs.2018.05.198.

## How to set up and run the code
1. Download the codes from this repository.
2. Upload the codes to Jupyter Notebook or Google CoLab
3. Click "Run all cells"

Remark: It is strongly recommended using GPU Farm to reduce the execution time.

## Degree of code reproduction
### What have they done:
1. Start from pre-trained CNNs (AlexNet, GoogLeNet/Inception, ResNet50).
2. Replace the last three layers with a new fully connected layer, a softmax layer, and a classification output layer sized to the target classes.
3. Increase the learning rate factors for the final fully connected layer to train faster.
4. Train the networks with GPU-appropriate settings on the target datasets (CIFAR-100, CIFAR-10, ImageNet subsets).
5. Test with testing set of target datasets (CIFAR-100, CIFAR-10, ImageNet subsets) and video.

### Settings mentioned in the paper
Input image sizes:
1. AlexNet: 224 x 224
2. GoogLeNet: 227 x 227
3. ResNet50: 227 x 227

Datasets:
1. CIFAR-100: 50000 training, 10000 testing
2. CIFAR-10: 50000 training, 10000 testing
3. ImageNet
4. Video(unknown source)

Layer customization:
Replaced the last three layers with:
1. A new fully connected (dense) layer
2. A softmax layer
3. A classification output layer

### My implementation
Remark: Due to the limited information provided by the paper, the missing hyperparameters are setted based on what is commonly used.

Learning rate strategy: 
1. Pre-trained layers: 0.001
2. Final fully connected layer: 0.01
   
Optimizer: Adam

Batch size: 30 (mini-batch size)

Number of epochs: 10
Input image sizes:
1. AlexNet: 224 x 224
2. GoogLeNet/ResNet50: 32x32 (modified due to long running time - see GoogLeNet_CIFAR10(227x227).ipynb)
   
Data augmentation: 
1. Random rotation: 30
2. Random horizontal flipping: 0.3
   
Loss function: Cross-entropy

Validation: 20%

Layer customization:
1. Only replace the last fully connected layer with a new fully connected layer (modified due to very poor performance of the orginal layer replacement strategy - GoogLeNet_CIFAR10(changed layer).ipynb)
2. The final fully connected layer size equals the number of target classes

Datasets:
1. CIFAR-100: 50000 training, 10000 testing
2. CIFAR-10: 50000 training, 10000 testing
3. ImageNet: cancelled as downloading of imagenet dataset is not available
4. Video: cancelled due to unknown source

### Description of each code
1. GoogLeNet_CIFAR10(changed layer).ipynb
This code follows the layer replacement strategy mentioned in the paper: replace the last three layers with a new fully connected layer, a softmax layer, and a classification output layer.
It takes the input size of 32x32 to shorten the running time.
Result: It gives very poor performance, so modification is made on future code implementation.
Future modification: Only replace the last fully connected layer with a new fully connected layer.

2. GoogLeNet_CIFAR10(227x227).ipynb
This code follows input size mentioned in the paper: 227 x 227.
Result: It takes very long time to run.
Future modification: Take input size of 32x32 for GoogLeNet and ResNet50.

3. GoogLeNet_CIFAR10(32x32).ipynb
This code takes the input size of 32x32 and only replace the last fully connected layer with a new fully connected layer.
Result: Fair performance (see more detail in the code)

4. GoogLeNet_CIFAR100.ipynb
This code takes the input size of 32x32 and only replace the last fully connected layer with a new fully connected layer.
Result: Poor performance (see more detail in the code)

5. ResNet50_CIFAR10.ipynb
This code takes the input size of 32x32 and only replace the last fully connected layer with a new fully connected layer.
Result: Fair performance (see more detail in the code)

6. ResNet50_CIFAR100.ipynb
This code takes the input size of 32x32 and only replace the last fully connected layer with a new fully connected layer.
Result: Poor performance (see more detail in the code)

7. AlexNet_CIFAR10.ipynb
This code takes the input size of 227x227 and only replace the last fully connected layer with a new fully connected layer.
Result: Good performance (see more detail in the code)

8. AlexNet_CIFAR100.ipynb
This code takes the input size of 227x227 and only replace the last fully connected layer with a new fully connected layer.
Result: Good performance (see more detail in the code)

## Author
Lo Tsz Yan
UID: 3036222276
