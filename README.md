# YOLO-Object-Counting-API
Real time Object Counting api. Implemented with the [YOLO](https://arxiv.org/pdf/1612.08242.pdf) algorithm and with the [SORT](https://arxiv.org/pdf/1703.07402.pdf) algorithm

The implementation is using model in same format as darkflow and darknet. Weight files, as well as cfg files can be found [here](http://pjreddie.com/darknet/yolo/). Darklow supports only YOLOv1 and YOLOv2. Support for YOLOv3 has not yet been implemented.

In order to achieve the best performance, you should have Cuda and tensorflow-gpu installed on Your device. 

## Demo
### Count objects of a specified class crossing a virtual line


#### Counting pedestrains 
<p align="center"> <img src="inputs/pedestrians_output.gif"/> </p>

#### Highway traffic counting

<p align="center"> <img src="inputs/highway_traffic_output.gif"/> </p>

### Count objects on a video

<p align="center"> <img src="inputs/count_objects_on_video.gif"/> </p>

### Count objects on a single frame

<p align="center"> <img src="inputs/count_people_output.jpg"/> </p>


# Set up
## Dependencies

```
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
    pip3 install -e .
    ```

3. Install with pip globally
    ```
    pip3 install .
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

## Run counting

Once You have all dependencies instaled and all required files You can start counting objects. Object counting is carried out by an ObjectCuntingAPI object.

Examples of counting below

### Count cars on crosing a virtual line 
```bash
python3 count_cars_crosing_virtual_line.py
```

### Count objects on video from Video Camera 
```bash
python3 count_objects_from_camera.py
```

### Count people on image 
```bash
python3 count_people_on_image.py
```

## Credits
The following open source projects were used in the implementation

### Darkflow

The YOLO algorithm impementation  - [Darkflow](https://github.com/thtrieu/darkflow)

### Python Traffic Counter

[Object counting with YOLO and SORT](https://github.com/bharath5673/python-traffic-counter-with-yolo-and-sort). Similar project, but instead of using the darklow YOLO implementation, it uses the opencv YOLO implementation, so there is no GPU acceleration.

### Deep Sort 
Object tracking and counting - [SORT](https://github.com/abewley/sort)

## Images and Videos sources
Highway surveillance [video](https://www.youtube.com/watch?v=PJ5xXXcfuTc)

Pedestrian surveillance [video](https://www.youtube.com/watch?v=aUdKzb4LGJI)

## Authors
* [tugot17](https://github.com/tugot17)

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details


That's all.
