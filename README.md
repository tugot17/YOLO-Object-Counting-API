# YOLO-Object-Counting-API
Real time Object Counting api. Implemented with the [YOLO](https://arxiv.org/pdf/1612.08242.pdf) algorithm and with the [SORT](https://arxiv.org/pdf/1703.07402.pdf) algorithm

The implementation is using model in same format as darkflow and darknet. Weight files, as well as cfg files can be found [here](http://pjreddie.com/darknet/yolo/). Darklow supports only YOLOv1 and YOLOv2. Support for YOLOv3 has not yet been implemented.

In order to achieve the best performance, you should have Cuda and tensorflow-gpu installed on Your device. 

## Demo
### Count objects of a specified class crossing a virtual line
See demo below or see on [this imgur](http://i.imgur.com/EyZZKAA.gif)

<p align="center"> <img src="demo.gif"/> </p>

### Count objects on a video

### Count objects on a single frame

# Set up
## Dependencies

```
-Python3
-tensorflow 1.0
-numpy
-opencv 3
```

## Getting started

You can choose _one_ of the following three ways to get started with darkflow.

1. Just build the Cython extensions in place. NOTE: If installing this way you will have to use `./flow` in the cloned darkflow directory instead of `flow` as darkflow is not installed globally.
    ```
    python3 setup.py build_ext --inplace
    ```

2. Let pip install darkflow globally in dev mode (still globally accessible, but changes to the code immediately take effect)
    ```
    pip install -e .
    ```

3. Install with pip globally
    ```
    pip install .
    ```

## Required files

The YOLO algoritym impementation used in this project requires 3 files. Configuration of network (.cfg), trained weights (.weights) and labels.txt. 

YOLO implementation used in this project enables usage of YOLOv1 and YOLOv2, and its tiny versions. Support for YOLOv3 has not yet been implemented.


### .cfg files
Configuration file determines a network architecture. Configurations can be found [here](http://pjreddie.com/darknet/yolo/). In example scripts we assume that the configuration is placed in cfg/ folder. Location of used .cfg file is specyfied in the options object used in the code. 

The .cfg file can be downloaded using the following command: 
```bash
wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov2.cfg -O cfg/yolov2.cfg
```
### .weights files
The .weights files contain trained parameters of a network. In example scripts we assume the weights are placed in bin/ folder. Location of used .weights file is specyfied in the options object used in the code.

The .weights file can be downloaded using the following command:
```bash
wget https://pjreddie.com/media/files/yolov2.weights -O bin/yolov2.weights
```
### labels.txt files

This file is list of classes detected by a YOLO netowork. It shoud contain as many classes as it is specyfied in a .cfg file. 



## Credits

The YOLO Object counting API, is based on the YOLO and SORT algorithms. In this project as an YOLO implementation we use darkflow

## Authors
* [tugot17](https://github.com/tugot17)

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE.md](LICENSE.md) file for details


That's all.
