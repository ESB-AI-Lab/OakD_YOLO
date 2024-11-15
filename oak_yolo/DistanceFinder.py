import cv2
import depthai as dai
import numpy as np
from ultralytics import YOLO
import time
from datetime import timedelta
import torch
import yaml

class DistanceFinder:
	"""
	This class represents a distance finder 
	
	Uses depth data and object detections to find the depth of objects
	"""
	
	def __init__(self,settings: str):
		"""
		Initializes a DataCollector object
		
		Args:
			settings (str): Path to YAML file for settings
		"""
		# Settings
		self.settings_path = settings
		self.model = None
		self.device = "cpu"
		self.preview:bool = True
		self.camera_settings:dict = {"fps":60,"stereoRes":400,"previewRes":(1280,720),
										"floodLightIntensity":1,"laserDotProjectorIntensity":1}
		# Pipelines For Camera
		self.pipeline = dai.Pipeline()
		self.rgb = self.pipeline.create(dai.node.Camera)
		self.rgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
		self.left = self.pipeline.create(dai.node.MonoCamera)
		self.left.setBoardSocket(dai.CameraBoardSocket.LEFT)
		self.right = self.pipeline.create(dai.node.MonoCamera)
		self.right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
		self.stereo = self.pipeline.create(dai.node.StereoDepth)
		self.sync = self.pipeline.create(dai.node.Sync)
		self.sync.setSyncThreshold(timedelta(milliseconds=50))
		self.spatialLocationCalculator = self.pipeline.create(dai.node.SpatialLocationCalculator)
		self.spatialData = self.pipeline.create(dai.node.XLinkOut)
		self.spatialCalcConfig = self.pipeline.create(dai.node.XLinkIn)
		# Stereo Depth Config
		self.stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
		self.stereo.setLeftRightCheck(True)
		self.stereo.setSubpixel(True)
		self.stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
		# Output Pipelines
		self.sync_out = self.pipeline.create(dai.node.XLinkOut)
		self.sync_out.setStreamName("sync")
		self.spatialData.setStreamName("spatialData")
		self.spatialCalcConfig.setStreamName("spatialCalcConfig")
		# Spatial Location Calculator Config
		self.spatialLocationCalculator.inputConfig.setWaitForMessage(False)
		topLeft = dai.Point2f(0.4, 0.4)
		bottomRight = dai.Point2f(0.6, 0.6)
		config = dai.SpatialLocationCalculatorConfigData()
		config.depthThresholds.lowerThreshold = 100
		config.depthThresholds.upperThreshold = 10000
		config.calculationAlgorithm = dai.SpatialLocationCalculatorAlgorithm.MEDIAN
		config.roi = dai.Rect(topLeft, bottomRight)
		self.spatialLocationCalculator.initialConfig.addROI(config)
		# Linking
		self.rgb.preview.link(self.sync.inputs["rgbSync"])
		self.stereo.depth.link(self.sync.inputs["depthSync"])
		self.left.out.link(self.stereo.left)
		self.right.out.link(self.stereo.right)
		self.sync.out.link(self.sync_out.input)
		self.stereo.depth.link(self.spatialLocationCalculator.inputDepth)
		self.spatialLocationCalculator.passthroughDepth.link(self.sync.inputs["depthSync"])
		self.spatialLocationCalculator.out.link(self.spatialData.input)
		self.spatialCalcConfig.out.link(self.spatialLocationCalculator.inputConfig)
		# Call Setup Function
		self.__setup()
		self.__camera_setup()
		
	def __setup(self):
		"""
		Helper function YAML parsing
		"""
		# Process YAML Settings File
		with open(self.settings_path, 'r') as file:
			data = yaml.safe_load(file)
			if 'data' in data:
				data_params = data['data']
				for item in data_params:
					if item=="model":
						self.model = YOLO(data_params["model"])
					if item=="device":
						self.device=data_params["device"]
					if item=="preview":
						self.preview=bool(data_params["preview"])
			if 'camera' in data:
				camera_params = data['camera']
				for item in camera_params:
					if item=="fps":
						self.camera_settings["fps"]=int(camera_params["fps"])
					if item=="stereoRes":
						self.camera_settings["stereoRes"]=int(camera_params["stereoRes"])
					if item=="previewRes":
						x,y = camera_params["previewRes"].split("x")
						self.camera_settings["previewRes"]=(int(x),int(y))
					if item=="floodLightIntensity":
						self.camera_settings["floodLightIntensity"]=int(camera_params["floodLightIntensity"])
					if item=="laserDotProjectorIntensity":
						self.camera_settings["laserDotProjectorintensity"]=int(camera_params["laserDotProjectorIntensity"])
			if not self.model:
				self.model = YOLO("yolo11n.pt")
			
	def __camera_setup(self):
		"""
		Helper function for camera settings
		"""
		for key in self.camera_settings:
			if key=="fps":
				self.rgb.setFps(self.camera_settings["fps"])
				self.left.setFps(self.camera_settings["fps"])
				self.right.setFps(self.camera_settings["fps"])
			if key=="stereoRes":
				stereo_res = None
				stereo_res_input=self.camera_settings["stereoRes"]
				if stereo_res_input==720:
					stereo_res = dai.MonoCameraProperties.SensorResolution.THE_720_P
				elif stereo_res_input==800:
					stereo_res = dai.MonoCameraProperties.SensorResolution.THE_800_P
				elif stereo_res_input==400:
					stereo_res = dai.MonoCameraProperties.SensorResolution.THE_400_P
				elif stereo_res_input==1200:
					stereo_res = dai.MonoCameraProperties.SensorResolution.THE_1200_P
				else:
					raise ValueError("Improper Stereo Respolution")
				self.left.setResolution(stereo_res)
				self.right.setResolution(stereo_res)
			if key=="previewRes":
				self.rgb.setPreviewSize(self.camera_settings["previewRes"][0],self.camera_settings["previewRes"][1])

	def __roi_config(self,frame,x1,y1,x2,y2):
		"""
		Function to generate data for Spatial Location Calculator
		"""
		# Define the ROI for spatial calculation
		topLeft = dai.Point2f(x1 / frame.shape[1], y1 / frame.shape[0])
		bottomRight = dai.Point2f(x2 / frame.shape[1], y2 / frame.shape[0])
        # Define Config Data
		config = dai.SpatialLocationCalculatorConfigData()
		config.depthThresholds.lowerThreshold = 100
		config.depthThresholds.upperThreshold = 10000
		config.calculationAlgorithm = dai.SpatialLocationCalculatorAlgorithm.MEDIAN
		config.roi = dai.Rect(topLeft, bottomRight)
		return config 

	def start(self):
		"""
		Function to start distance finder
		"""
		# Connect To Device And Start Pipeline
		with dai.Device(self.pipeline,maxUsbSpeed=dai.UsbSpeed.SUPER) as device:
			# Add Device Settings
			device.setIrFloodLightIntensity(self.camera_settings["floodLightIntensity"])
			device.setIrLaserDotProjectorIntensity(self.camera_settings["laserDotProjectorIntensity"])
			synced = device.getOutputQueue(name="sync",maxSize=1,blocking=False)
			spatialCalcQueue = device.getOutputQueue(name="spatialData", maxSize=1, blocking=False)
			spatialCalcConfigInQueue = device.getInputQueue("spatialCalcConfig")
			while True:
				# Get Frames From The Camera
				syncData = synced.get()
				inDepth = None
				inRGB = None
				for name,msg in syncData:
					if name == "depthSync":
						inDepth = msg
					if name == "rgbSync":
						inRGB = msg
				depthFrame = inDepth.getFrame()
				rgbFrame = inRGB.getCvFrame()
				depthFrameColorized = cv2.applyColorMap(cv2.convertScaleAbs(depthFrame, alpha=0.03), cv2.COLORMAP_JET)
		    	# Get Model Predictions
				predictions = self.model(rgbFrame, device=torch.device(self.device))
				if len(predictions[0].boxes)==0:
					# Display Preview
					if self.preview:
						combined = np.concatenate([depthFrameColorized,rgbFrame],axis=0)
						cv2.imshow("Combined",combined)
						if cv2.waitKey(1) == ord('q'):
		        					break
					continue
				# Iterate Through YOLO Detections And Get Updated ROIs
				for detection in predictions:
					roi_list=[]
					for box in detection.boxes:
						x1, y1, x2, y2= box.xyxy[0]
						x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
						cls = box.cls.item()
						cls = self.model.names[cls]
						roi_list.append(self.__roi_config(depthFrame,x1,y1,x2,y2))
						cv2.putText(rgbFrame,cls,(x1+10,y1+20),cv2.FONT_HERSHEY_TRIPLEX, 0.5 , (0,0,255))
				# Add ROIs To Config Queue And Get Spatial Data
				cfg = dai.SpatialLocationCalculatorConfig()
				cfg.setROIs(roi_list)
				spatialCalcConfigInQueue.send(cfg)	
				spacialData = spatialCalcQueue.get().getSpatialLocations()
				# Iterate Through Spatial Data
				for object in spacialData:
					roi = object.config.roi
					roi = roi.denormalize(width=1280, height=720)
					xmin = int(roi.topLeft().x)
					ymin = int(roi.topLeft().y)
					xmax = int(roi.bottomRight().x)
					ymax = int(roi.bottomRight().y)
					color=(0,0,255)
					fontType = cv2.FONT_HERSHEY_TRIPLEX
					cv2.rectangle(rgbFrame, (xmin, ymin), (xmax, ymax), color, 1)
					cv2.putText(rgbFrame, f"X: {int(object.spatialCoordinates.x)} mm", (xmin + 10, ymin + 35), fontType, 0.5, color)
					cv2.putText(rgbFrame, f"Y: {int(object.spatialCoordinates.y)} mm", (xmin + 10, ymin + 50), fontType, 0.5, color)
					cv2.putText(rgbFrame, f"Z: {int(object.spatialCoordinates.z)} mm", (xmin + 10, ymin + 65), fontType, 0.5, color)
				# Display Preview
				if self.preview:
					combined = np.concatenate([depthFrameColorized,rgbFrame],axis=0)
					cv2.imshow("Combined",combined)
					if cv2.waitKey(1) == ord('q'):
						break 
		cv2.destroyAllWindows()
			



