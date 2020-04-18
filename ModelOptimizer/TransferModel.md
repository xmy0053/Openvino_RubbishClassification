## How to transfer your ONNX model to IR model which can be deployed on OpenVINO?
If you have a ONNX model and OpenVINO environment on Win10 platform, you can transfer the ONNX file by following steps;
1. Run Powershell/CMD with administrator
2. change your working path to the OpenVINO model_optimizer folder, following command use default path;
> cd "C:\Program Files (x86)\IntelSWTools\openvino_2019.3.379\deployment_tools\model_optimizer"
3. Copy your ONNX File to current working path
> cp G:\RubbishClassfication\Train_Pytorch\ResNet.onnx
4. make new folder for the new model
>mkdir Output_IR_Model
5. Run Python script mo.py (Tips: If you want to deploy your model on NCS or NCS2, you must add option --data_type FP16)
>python mo.py --input_model ResNet.onnx --data_type FP16 --output_dir Output_IR_Model
