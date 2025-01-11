# Object Distance Finder with OakD and YOLO
Run YOLO object detection models on Jetson Nano and use [depthai library](https://github.com/luxonis/depthai/tree/main) and [depthai-experiments ](https://github.com/luxonis/depthai-experiments/tree/master) to obtain depth data. Run script and [Docker image](https://github.com/users/Kevin12J/packages/container/package/oakd_yolo) built from [dusty-nv/jetson-container](https://github.com/dusty-nv/jetson-containers)<sup>1</sup> repo. All code tested on a Jetson Orin Nano with Luxonis Oak-D Pro W.
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
  outputPath: # path to output data (str)
  outputInterval: # interval of frame to save data(int)
camera:
  fps: # frames per second (int)
  stereoRes: # stereo camera respolution (400,720,800,1200)
  previewRes: # resolution of rgb camera (widthxheight)
  floodLightIntensity: # flood light intensity (0 to 1)
  laserDotProjectorIntensity: # laser dot projector intensity (0 to 1)
```
#### Usage
Initialize `DistanceFinder` object and call `start` function to initialize device connection and use `get_frame` function to perform detections and depth calculations on each frame
```
from oak_yolo.DistanceFinder import DistanceFinder

finder = DistanceFinder("settings.yaml")
finder.start()
while(True):
  data = finder.get_frame()
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
Obtain coordinates of detected objects from `Data` object using `getSpatials` function
```
image = data[0].getSpatials()
```
Visualize depth and image frames from `Data` object using `visualize` function and pass in `draw_boxes=True` to include bouding boxes in visualization
```
for item in Data:
  item.visualize(draw_boxes=True)
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
