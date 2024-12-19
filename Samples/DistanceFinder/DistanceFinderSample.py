from oak_yolo.DistanceFinder import DistanceFinder

finder = DistanceFinder("settings.yaml")
print(finder)
finder.start()

while(True):
	data = finder.get_frame()
