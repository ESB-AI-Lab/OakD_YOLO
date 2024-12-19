# Object Distance Finder with OakD and YOLO
Run YOLO object detection models on Jetson Nano and use [depthai library](https://github.com/luxonis/depthai/tree/main) to obtain depth data. Run script and [Docker image](https://github.com/users/Kevin12J/packages/container/package/oakd_yolo) built from [dusty-nv/jetson-container](https://github.com/dusty-nv/jetson-containers)<sup>1</sup> repo. All code tested on a Jetson Orin Nano with Luxonis Oak-D Pro W.
## Installation and Setup
Clone repo
```
git clone https://github.com/ESB-AI-Lab/OakD_YOLO.git
```
Run setup script to link module to samples
```
cd OakD_YOLO
bash setup.sh
```
See [samples](#Samples) to run samples
## oak_yolo module
### Distance Finder
Run YOLO model and obtain depth information of detected objects
#### Settings
Below are the settings options to be put in `yaml` file
```
data:
  model: # path to model (str)
  device: # device to perform inference on ("cpu","cuda")
  preview: # video preview (bool)
camera:
  fps: # frames per second (int)
  stereoRes: # stereo camera respolution (400,720,800,1200)
  previewRes: # resolution of rgb camera (widthxheight)
  floodLightIntensity: # flood light intensity (0 to 1)
  laserDotProjectorIntensity: # laser dot projector intensity (0 to 1)
```
#### Usage
Initialize `DistanceFinder` object and call `start` function to perform detections and depth calculations
```
from oak_yolo.DistanceFinder import DistanceFinder

finder = DistanceFinder("settings.yaml")
finder.start()
```
***
### Data Collector
Collect depth frames and images to gather sample data
#### Settings
Below are the settings options to be put in `yaml` file
```
data:
  output: # output directory path (str)
  interval: # interval to save frame (int)
  preview: # video preview (bool)
camera:
  fps: # frames per second (int)
  stereoRes: # stereo camera respolution (400,720,800,1200)
  previewRes: # resolution of rgb camera (widthxheight)
  floodLightIntensity: # flood light intensity (0 to 1)
  laserDotProjectorIntensity: # laser dot projector intensity (0 to 1)
```
#### Usage
Initialize `DataCollector` object and call `start` function to collect data. Make sure there is a symbolic link using `ln -s` to `oak_yolo`.
```
from oak_yolo.DataCollector import DataCollector

colector = DataCollect(path_to_settings.yaml)
collector.start()
```
***
### Data Loader
Load and interact with data from Data Collector
#### Settings
Below are the settings options to be put in `yaml` file
```
data:
  directory: # data directory path (str)
```
#### Usage
Initialize `DataLoader` object and call `load` to load data as a list of `Data` objects
```
from oak_yolo.DataLoader import DataLoader

data = DataLoader(settings="settings.yaml")
data.load()
```
Find the number of `Data` objects using `len` operator
```
print(len(data))
```
Access `Data` objects using list indexing
```
first = data[0]
every_other = data[::2]
```
Loop through `Data` objects with loops
```
for item in Data:
  curr = item

for i in range(len(data)):
  curr = data[i]
```
Obtain depth data as numpy array from `Data` object using `getDepth` function
```
depth = data[0].getDepth()
```
Obtain image data using `cv2.imread` from `Data` object using `getImage` function
```
image = data[0].getImage()
```
Visualize depth and image frames from `Data` object using `visualize` function
```
for item in Data:
  item.visualize()
```
## Samples
Start Docker container with run script and use `-v` argument to mount `OakD_YOLO` directory.
```
bash run.sh -v path/OakD_YOLO:/home
```
Once inside container go to ```Samples``` directory
```
cd home/Samples
```
There will be three directories for each part of oak_yolo module which have a `.py` file and `settings.yaml` file.
___
<span style="font-size: 0.005em;"><sup>1</sup>Franklin, D. Jetson Containers(Machine Learning Containers for Jetson and JetPack) [Computer software]. https://github.com/dusty-nv/jetson-containers</span>
