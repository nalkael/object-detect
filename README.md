# Object Detection Model for Urban Infrastructure

### Introduction
This project is used in detection for **urban infrastructures** from input of ***[Orthomosaic](https://www.dronegenuity.com/orthomosaic-maps-explained/) images*** , more precisely,  small infrastructures on the street such as the ***manhole cover***, ***utility shaft***, ***water valve cover***, ***gas valve cover***, ***underground hydrant***, ***stormwater inlet***, etc. 

Here are some example of  

The trained model can run on the local computer without additional computation resource and detect classified objects from arbitrary input images.

And here is a flow chart for the pipeline:

```mermaid
graph LR
subgraph Training Stage
A(Traning Data) 
A --> B((Base Model))
B -.- C((train)) -.-> B
end
B --> D(Trained Model)
subgraph Test Stage
E(Test Data) --> D
D --> F(Object Coordinates)
D --> G(Object Classifications)
end
```

### Preparing your environment

Get yourself a Python>=3.10 environment. Using a  [virtualenv](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/#creating-a-virtual-environment)  is recommended but not required.

You'll need a few tools to run scripts in this distribution. They are specified in the requirements.txt file. Install them with pip:

> python -m pip install -r requirements.txt

and also install [Detectron2](https://github.com/facebookresearch/detectron2):
>git clone https://github.com/facebookresearch/detectron2.git
>python -m pip install -e detectron2

This project contains four basic models (could extend more in future):
- Faster R-CNN
- RetinaNet
- SSD
- YOLOv5

### TODO
