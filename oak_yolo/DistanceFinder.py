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
from oak_yolo.CameraInterface import CameraInterface
from oak_yolo.OakD_Cam import OakD_Cam
from oak_yolo.OakD_Gazebo import OakD_Gazebo

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
		self.camera:CameraInterface = None
		# Call Setup Function
		self.__setup()
		
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
				if camera_params["sim"]:
					self.camera = OakD_Gazebo(width=640,height=480,HFOV=1.274)
				else:
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
					self.camera = OakD_Cam(camera_settings=self.camera_settings)
			if not self.model:
				self.model = YOLO("yolo11n.pt",task="detect")

	def start(self):
		"""
		Function to start distance finder
		"""
		# Initialize Spatial Calculator and start Camera	
		self.spatialCalculator = HostSpatialsCalc(self.camera)
		self.camera.start()
		# Create Directories For Output
		if self.outputPath:
			if not os.path.exists(self.outputPath):
				os.mkdir(self.outputPath)
			if not os.path.exists(self.outputPath+"/depth"):
				os.mkdir(self.outputPath+"/depth")
			if not os.path.exists(self.outputPath+"/images"):
				os.mkdir(self.outputPath+"/images")
			if not os.path.exists(self.outputPath+"/spatials"):
				os.mkdir(self.outputPath+"/spatials")
		
	def process_frame(self):
		# Array To Store Object Data
		data:dict = {}
		# Gete frames from camera
		frame = self.camera.get_frame()
		depthFrame = frame["depth"]
		rgbFrame = frame["rgb"]
		data["depth"] = depthFrame
		data["rgb"] = rgbFrame
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
		objects = []
		for detection in predictions:
			for box in detection.boxes:
				x1, y1, x2, y2 = box.xyxy[0]
				x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
				object_class = box.cls.item()
				object_class = self.model.names[object_class]
				spatials, center = self.spatialCalculator.calc_spatials(depthFrame,[x1,y1,x2,y2],averaging_method=np.median)
				objects.append({"box":(x1,y1,x2,y2),"class":object_class,"depth":int(spatials['z'])})
				color=(0,0,255)
				fontType = cv2.FONT_HERSHEY_TRIPLEX
				cv2.putText(rgbFrame,object_class,(x1+10,y1+20),cv2.FONT_HERSHEY_TRIPLEX, 0.5 , color)
				cv2.rectangle(rgbFrame, (x1, y1), (x2, y2),color, 1)
				cv2.putText(rgbFrame, f"X: {int(spatials['x'])} mm", (x1 + 10, y1 + 35), fontType, 0.5, color)
				cv2.putText(rgbFrame, f"Y: {int(spatials['y'])} mm", (x1 + 10, y1 + 50), fontType, 0.5, color)
				cv2.putText(rgbFrame, f"Z: {int(spatials['z'])} mm", (x1 + 10, y1 + 65), fontType, 0.5, color)
		data["objects"] = objects
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


