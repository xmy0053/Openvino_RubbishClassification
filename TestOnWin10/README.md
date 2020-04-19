## classfication_sample.py
**This is the test file on Win10**


This is the example from the openvino. I modified slightly since train model using RGB format images, while Opencv read image with RGB format by default. Meanwhile, the 8 bit image [0,255] should be normalized to [-0.5,0.5] FP16.
You can find the original test file in the path of Openvino for the reference. Run script using following command.
> python3 .\classification_sample.py -m ResNet.xml -i [Path to a folder with images or path to an image files]
