import cv2
import depthai as dai
import os
from datetime import timedelta
import numpy as np
import yaml

class DataCollector:
	"""
	This class represents a DataCollector for the OakD Camera.
	
	Stores depth frames as numpy arrays and camera images.
	"""
	
	def __init__(self,settings: str):
		"""
		Initializes a DataCollector object
		
		Args:
			settings (str): Path to YAML file for settings
		"""
		# Settings
		self.settings_path = settings
		self.output_path:str = "./Data"
		self.preview:bool = True
		self.data_interval:int = 1
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
					if item=="output":
						self.output_path=data_params["output"]
					if item=="interval":
						self.data_interval=int(data_params["interval"])
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
		# Create Directories For Output
		if not os.path.exists(self.output_path):
			os.mkdir(self.output_path)
		if not os.path.exists(self.output_path+"/depth"):
			os.mkdir(self.output_path+"/depth")
		if not os.path.exists(self.output_path+"/images"):
			os.mkdir(self.output_path+"/images")
			
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
		Function to start data collection.
		"""
		# Connect To Device And Start Pipeline
		with dai.Device(self.pipeline,maxUsbSpeed=dai.UsbSpeed.SUPER) as device:
			# Add Device Settings
			device.setIrFloodLightIntensity(self.camera_settings["floodLightIntensity"])
			device.setIrLaserDotProjectorIntensity(self.camera_settings["laserDotProjectorIntensity"])
			synced = device.getOutputQueue(name="sync",maxSize=1,blocking=True)
			count = 0
			while True:
				# Get Data From Sync Queue
				syncData = synced.get()
				inDepth = None
				inRGB = None
				for name,msg in syncData:
					if name=="depthSync":
						inDepth = msg
					if name=="rgbSync":
						inRGB = msg
				# Get Frames
				depthFrame = inDepth.getFrame()
				rgbFrame = inRGB.getCvFrame()
				# Save Frame/Data
				if(count%self.data_interval==0):
					print(f'Saving Data # {count//self.data_interval}')
					cv2.imwrite(f'{self.output_path}/images/RGB_{count//self.data_interval}.jpg',rgbFrame)
					np.save(f'{self.output_path}/depth/DEPTH_{count//self.data_interval}',depthFrame)
				# Show Output
				if(self.preview):
					depthFrameColorized = cv2.applyColorMap(cv2.convertScaleAbs(depthFrame, alpha=0.03), cv2.COLORMAP_JET)
					combined = np.concatenate([depthFrameColorized,rgbFrame],axis=0)
					cv2.imshow("Combined",combined)
					if cv2.waitKey(1) == ord('q'):
						break
				count+=1
			cv2.destroyAllWindows()
