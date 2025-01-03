import cv2
import depthai as dai
import numpy as np
from ultralytics import YOLO
import time
from datetime import timedelta
import torch
import yaml
import json
import os
from oak_yolo.calc import HostSpatialsCalc 

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
		self.outputPath = None
		self.outputInterval = 1
		self.frameCount = 0
		self.camera_settings:dict = {"fps":60,"stereoRes":400,"previewRes":(1280,720),
										"floodLightIntensity":1,"laserDotProjectorIntensity":1}
		# Camera
		self.camera = None
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
		# Spatial Calculator
		self.spatialCalculator = None
		# Stereo Depth Config
		self.stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
		self.stereo.setLeftRightCheck(True)
		self.stereo.setSubpixel(True)
		self.stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
		# Output Pipelines
		self.sync_out = self.pipeline.create(dai.node.XLinkOut)
		self.sync_out.setStreamName("sync")
		# Linking
		self.rgb.preview.link(self.sync.inputs["rgbSync"])
		self.stereo.depth.link(self.sync.inputs["depthSync"])
		self.left.out.link(self.stereo.left)
		self.right.out.link(self.stereo.right)
		self.sync.out.link(self.sync_out.input)
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
						self.model = YOLO(data_params["model"],task="detect")
					if item=="device":
						self.device=data_params["device"]
					if item=="preview":
						self.preview=bool(data_params["preview"])
					if item=="outputPath":
						self.outputPath=data_params["outputPath"]
					if item=="outputInterval":
						self.outputInterval=int(data_params["outputInterval"])
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
				self.model = YOLO("yolo11n.pt",task="detect")
			
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

	def start(self):
		"""
		Function to start distance finder
		"""
		# Connect To Device And Start Pipeline
		self.camera = dai.Device(self.pipeline,maxUsbSpeed=dai.UsbSpeed.SUPER)
		# Add Device Settings
		self.camera.setIrFloodLightIntensity(self.camera_settings["floodLightIntensity"])
		self.camera.setIrLaserDotProjectorIntensity(self.camera_settings["laserDotProjectorIntensity"])
		self.synced = self.camera.getOutputQueue(name="sync",maxSize=1,blocking=False)
		self.spatialCalculator = HostSpatialsCalc(self.camera)
		# Create Directories For Output
		if not os.path.exists(self.outputPath):
			os.mkdir(self.outputPath)
		if not os.path.exists(self.outputPath+"/depth"):
			os.mkdir(self.outputPath+"/depth")
		if not os.path.exists(self.outputPath+"/images"):
			os.mkdir(self.outputPath+"/images")
		if not os.path.exists(self.outputPath+"/spatials"):
			os.mkdir(self.outputPath+"/spatials")
		
	def get_frame(self):
		# Array To Store Object Data
		data = []
		# Get Frames From The Camera
		syncData = self.synced.get()
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
		# Save RGB/Depth
		if(self.outputPath and self.frameCount%self.outputInterval==0):
			cv2.imwrite(f'{self.outputPath}/images/RGB_{self.frameCount//self.outputInterval}.jpg',rgbFrame)
			np.save(f'{self.outputPath}/depth/DEPTH_{self.frameCount//self.outputInterval}',depthFrame)
		# Get Model Predictions
		predictions = self.model(rgbFrame, device=torch.device(self.device), verbose=False)
		if len(predictions[0].boxes)==0:
			# Display Preview
			if self.preview:
				combined = np.concatenate([depthFrameColorized,rgbFrame],axis=0)
				cv2.imshow("Combined",combined)
				cv2.waitKey(1)
			# Save Spatials
			if(self.outputPath and self.frameCount%self.outputInterval==0):
				with open(f'{self.outputPath}/spatials/SPATIAL_{self.frameCount//self.outputInterval}.json','w') as spatial_out:
					json.dump(data,spatial_out)
			self.frameCount+=1
			return data
		# Iterate Through YOLO Detections And Get Object Depths
		for detection in predictions:
			for box in detection.boxes:
				x1, y1, x2, y2 = box.xyxy[0]
				x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
				object_class = box.cls.item()
				object_class = self.model.names[object_class]
				spatials, center = self.spatialCalculator.calc_spatials(inDepth,[x1,y1,x2,y2],averaging_method=np.median)
				data.append({"box":(x1,y1,x2,y2),"class":object_class,"depth":int(spatials['z'])})
				color=(0,0,255)
				fontType = cv2.FONT_HERSHEY_TRIPLEX
				cv2.putText(rgbFrame,object_class,(x1+10,y1+20),cv2.FONT_HERSHEY_TRIPLEX, 0.5 , color)
				cv2.rectangle(rgbFrame, (x1, y1), (x2, y2),color, 1)
				cv2.putText(rgbFrame, f"X: {int(spatials['x'])} mm", (x1 + 10, y1 + 35), fontType, 0.5, color)
				cv2.putText(rgbFrame, f"Y: {int(spatials['y'])} mm", (x1 + 10, y1 + 50), fontType, 0.5, color)
				cv2.putText(rgbFrame, f"Z: {int(spatials['z'])} mm", (x1 + 10, y1 + 65), fontType, 0.5, color)
		# Display Preview
		if self.preview:
			combined = np.concatenate([depthFrameColorized,rgbFrame],axis=0)
			cv2.imshow("Combined",combined)
			cv2.waitKey(1)
		# Save Spatials
		if(self.outputPath and self.frameCount%self.outputInterval==0):
			with open(f'{self.outputPath}/spatials/SPATIAL_{self.frameCount//self.outputInterval}.json','w') as spatial_out:
				json.dump(data,spatial_out)
		self.frameCount+=1
		return data
		#cv2.destroyAllWindows()


