import numpy as np
import os
import cv2
import yaml
from oak_yolo.calc import HostSpatialsCalc 

class DataLoader:
	"""
	This class represents a DataLoader for DataCollector
	
	Load depth frames and images stored from DataCollector
	"""
	
	class Data:
		"""
		Inner class to represent each depth frame/image pair
		"""
		def __init__(self,depth,image,index):
			self.depth = depth
			self.image = image
			self.index = index
			
		def getDepth(self):
			"""
			Get numpy array for depth in data
			"""
			return np.load(self.depth)
		
		def getImage(self):
			"""
			Get image in data
			"""
			return cv2.imread(self.image)
			
		def getSpatials(self):
			"""
			Get coordinates of objects
			"""
			# Continue here
			return None 
			
		def visualize(self):
			"""
			Visualize data
			"""
			depthFrame = self.getDepth()
			imageFrame = self.getImage()
			depthFrameColorized = cv2.applyColorMap(cv2.convertScaleAbs(depthFrame, alpha=0.03), cv2.COLORMAP_JET)
			combinedFrames = np.concatenate([depthFrameColorized,imageFrame],axis=0)
			while True:
				cv2.imshow(str(self.index),combinedFrames)
				if cv2.waitKey(1) == 32:
					break
			cv2.destroyAllWindows()
			
	
	def __init__(self,settings: str):
		"""
		Initialize a DataLoader object
		
		Args:
			settings (str): Path to YAML file for settings
		"""
		self.data = []
		self.length = 0
		# settings
		self.settings_path = settings
		self.data_path = None
		self.model = None
		self.distanceCalculations = False
		self.__setup()
		
	def __setup(self):
		"""
		Helper function YAML parsing
		"""
		with open(self.settings_path, 'r') as file:
			data = yaml.safe_load(file)
			if 'directory' in data:
				self.data_path=data["directory"]
				if(not os.path.exists(self.data_path)):
					raise FileNotFoundError("Data Directory Doesn't Exist")
			if "model" in data:
				self.model=data["model"]
				
	def load(self):
		"""
		Function To Load Data
		"""
		depth_count=0
		depth_paths = os.listdir(self.data_path+"/depth")
		depth_paths.sort(key=lambda x:int(x.split(".")[0][-1]))
		image_paths = os.listdir(self.data_path+"/images")
		image_paths.sort(key=lambda x:int(x.split(".")[0][-1]))
		if len(depth_paths)!=len(image_paths):
			raise Exception("Data Error: depth and image count are not the same")
		for i in range(len(depth_paths)):
			self.data.append(self.Data(self.data_path+"/depth/"+depth_paths[i],self.data_path+"/images/"+image_paths[i],i))	
			self.length+=1
		
	def __iter__(self):
		"""
		Iterator For Class
		"""
		self._iter_index=0
		return self
		
	def __next__(self):
		"""
		Iteration Step
		"""
		if self._iter_index < self.length:
			value = self.data[self._iter_index]
			self._iter_index += 1
			return value
		else:
			raise StopIteration
			
	def __getitem__(self,index):
		"""
		List Indexing
		"""
		if isinstance(index,slice):
			start = index.start
			stop = index.stop
			step = index.step or 1
			pairs = []
			return self.data[star:stop:step]
		else:
			return self.data[index]
			
	def __len__(self):
		"""
		Length Attribute
		"""
		return self.length
		
			
			
