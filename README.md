# Openvino_RubbishClassification
```mermaid
1. Image -> The Folder contains images' source which are used in Markdown File;
2. ModelOptimizer -> Explain How to convert your ONNX file to INTEL IR Model via OpenVINO;
3. TestOnWin10 -> Test your IR Model on Win10;
4. Train -> Train a ONNX model using Pytorch
```
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

