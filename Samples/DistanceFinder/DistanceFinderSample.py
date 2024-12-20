from oak_yolo.DistanceFinder import DistanceFinder

finder = DistanceFinder("settings.yaml")
finder.start()

while(True):
	data = finder.get_frame()
	if(len(data)>0):
		print(data)
