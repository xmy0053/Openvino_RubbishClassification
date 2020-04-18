# Openvino_RubbishClassification
[TOC]

# Introduction

This project presents an AI driven Energy Efficient Recyclable Bin which sorts the recyclables into 4 types including metal, plastic, paper, and glass. We deploy Intel neural computing sticks(NCS) on embedded Linux systems to establish a complete embedded computer vision system. The image of the recyclable object is captured via camera, and will be classified and processed by the depth neural network. After processing, the operating system drives the servo mechanism to complete the classification of recyclables. Meanwhile, we also designed an energy collection system for the whole system, which is composed of solar panels, lithium batteries and power management system. Through this energy collection system, the above embedded system can be guaranteed to run without power grid.

## Environment Preparation 

### Deep Neural Network Training on Win10
* Anaconda
* Python 3.6.8
* Opencv-python 4.2.0
* CUDA 10.1
* Pytorch 

>**Note: Pytorch do not support CUDA 10.2 and NVIDIA website offers CUDA 10.2 by default**

### OpenVINO on Win10
* Openvino_2019.3.379
- [OpenCV 4.2](https://opencv.org/releases.html) or higher
- [Intel® C++ Compiler 2017 Update 4](https://software.intel.com/en-us/articles/redistributables-for-intel-parallel-studio-xe-2017-composer-edition-for-linux)
- [CMake* 3.4](https://cmake.org/download/) or higher
- [Python* 3.6.8](https://www.python.org/downloads/) 
- [Microsoft Visual Studio* 2015 or 2017](https://www.visualstudio.com/vs/older-downloads/)
- Reference:[Install Intel® Distribution of OpenVINO™ toolkit for Windows* 10](https://docs.openvinotoolkit.org/latest/_docs_install_guides_installing_openvino_windows.html)

>**Note: Openvino 2020.1 is not stable. I tried to use its Model optimizer for optimize the ONNX network, but it does not work.**

### Openvino On Raspberry

* [python 3.6.8](https://www.python.org/downloads/)
* Numpy
* Opencv 4.2
* [OpenVINO_2019.3.379 for Raspberry](https://docs.openvinotoolkit.org/latest/_docs_install_guides_installing_openvino_raspbian.html)
* **Note: OpenVINO on the different platforms should be same version, since we could only deploy the IR network which converted from Model Optimizer on the Inference Engine with the same version**

## Briefing of our AI framework

Deep neural networks are widely used in many aspects of our lives. However, in most applications, there are high requirements for the computing resources and power consumption of the platform. As a result, there are some ASCI chips specifically used for neural network deployment, such as Intel's Movidius neural computing sticks(NCS). Usually this kind of ASCI will be deployed on the embedded system as a coprocessor, responsible for numerous floating-point or fixed point number data operations. Through its supporting platform Openvino, Movidius has realized the conversion of various types of deep neural network format to Intermediate Representation (IR) of the network which can be deployed on various types of platforms such as INTEL CPU, GPU and Movidius. In this way, a good balance can be achieved between the power consumption of the platform and the performance of the neural network.

OpenVINO is mainly composed of two parts: Model Optimizer and Inference Engine. Model Optimizer process assumes you have a network model trained using a supported deep learning framework. The scheme below illustrates the typical workflow for deploying a trained deep learning model:

Model Optimizer produces an Intermediate Representation (IR) of the network, which can be read, loaded, and inferred with the Inference Engine.

![Work Flow of Openvino](\DocumentSuppport\Drawing1.png)

In this application, we first use the Pytorch framework on the PC platform to train the neural network on GPU, and export the trained network to the ONNX format. Model Optimizer will be used to convert the trained neural network into IR format. Copy the common format network to the Linux platform and deploy it on NCS through Linux's Inference Engine.


# ENGINEERING DESIGN

## Mechanical Design
The recyclable bin can be divided into 3 parts including identification area(Top), classification area(Middle), and storage area(bottle). Additionally, we attach one piece of solar panel to collect solar energy and support the operations of entity system.

1. Identification area
   >For the user, they will drop their trash in this area and the camera will capture the image which will identified by Raspberry Pi and its co-processor(NCS). Then, the system will drive servo-mechanical device according to the previous outcome.

2. Classification area
   >There are 2 servo-mechanical devices within this part, rigid plastic flap to keep trash waiting in last area and sorting bowl to establish one path to particular storage bin. Once captured image had been classified, the sorting bowl aligned to the targeted bin, and the trash released by removing plastic flap.    

3. Storage area

    >Storage area was split into several parts according to the number of classification.

## Hardware Design

![Hardware Design](/Image/Hardware.png)

The image illustrates the framework of the hardware design including power management and processing flow.  

### Power Management Unit

When it's sunny and brighten, the solar panel( $\approx$ 19.1V) collect the solar energy which charge the battery(7.2V~8.4V,Efficiency $\approx$ 80%) and transfer to the regulator(Efficency $\approx$ 90%) to supply the processing unit(5V). 

>Table 1 Power consumpation in IDLE Mode

|Device|Current(A)|Voltage(V)|Power(W)|
|:------|:---:|:-:|:-:|
| Raspiberry|0.5|5V|2.5|
| NCS|0.4|5V|2|
| Motor and driver|0.1|5|0.5|
| Battery(Charging)|1|7.4|7.4|

In most case, the system is work in IDLE mode to save energy and seldom will the system be triggered to sorting trashes. Hence, we can assess the power consumption by estimating current of each parts in IDLE which are shown in the Table 1.
$$
   P = E\sum {U_i \times I_i}
   =((2.5+2+0.5)\div0.9+7.4)\div0.8=16.1W
$$
The solar panel(19.1V) should output more than 0.85A current.

### Processing Unit
The processing unit consists of camera input, main processor, co-processor and motor drive. Firstly, the camera capture the image of recyclable and main processor cooperates with co-processor to classify the image and send particular signal to motor driver.

## Software Design
We use customized ResNet18 to classify recyclable images. The model was trained by Pytorch and transferred to ONNX model which can be recognized and optimized by Openvino model optimizer.

### Training a deep nerual network

### Model optimizing via Openvino

### Running test program on Win10

### Deploying on Raspberry Pi

**To be continued**
###   


