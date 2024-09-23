# Object Detection Model for Urban Infrastructure

### Introduction
This project is used in detection for **urban infrastructures** from input of ***[Orthomosaic](https://www.dronegenuity.com/orthomosaic-maps-explained/) images*** , more precisely,  small infrastructures on the street such as the ***manhole cover***, ***utility shaft***, ***water valve cover***, ***gas valve cover***, ***underground hydrant***, ***stormwater inlet***, etc. 

Here are some example of patterns for different objects:

![manhole cover](https://github.com/nalkael/object-detect/blob/main/examples/pattern%20sample/001_Schachtdeckel/001_Kanalschachtdeckel/KSr_01.jpg)![utility shaft](https://github.com/nalkael/object-detect/blob/main/examples/pattern%20sample/001_Schachtdeckel/002_Versorgungsschacht/VS_02.jpg)

![water valve cover](https://github.com/nalkael/object-detect/blob/main/examples/pattern%20sample/002_Schieberdeckel/001_Wasser/SD_Wasser_03.jpg)![gas valve cover](https://github.com/nalkael/object-detect/blob/main/examples/pattern%20sample/002_Schieberdeckel/002_Gas/SD_Gas_04.jpg)![underground hydrant](https://github.com/nalkael/object-detect/blob/main/examples/pattern%20sample/003_Unterflurhydrant/UFH_02.jpg)![stormwater inlet](https://github.com/nalkael/object-detect/blob/main/examples/pattern%20sample/004_Sinkkaesten/25x50/SK25_03.jpg)


The model can be trained on the local computer without additional huge computation resource and it can detect classified objects from arbitrary input images.

And here is a flow chart for the pipeline:

```mermaid
graph LR
subgraph Training Stage
A(Traning Data) 
A --> B((Base Model))
B -.-> C((train)) -.-> B
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
>
>python -m pip install -e detectron2

This project contains some foundation models such as:
- [Faster R-CNN](https://arxiv.org/abs/1506.01497)
- [YOLOv5](https://arxiv.org/html/2407.20892v1) (**Y**ou **O**nly **L**ook **O**nce)
- [SSD](https://arxiv.org/abs/1512.02325) (**S**ingle **S**hot Multibox **D**etector)
- [RetinaNet](https://arxiv.org/abs/1708.02002)
- [Cascade R-CNN](https://arxiv.org/abs/1712.00726)

### TODO


