from oak_yolo.DataLoader import DataLoader

data = DataLoader(settings="settings.yaml")
data.load()

for d in data:
	print(d.visualize())
	
print(len(data))
print(data[1])
