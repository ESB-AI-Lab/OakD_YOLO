from oak_yolo.DataLoader import DataLoader

data = DataLoader(settings="settings.yaml")
data.load()

for d in data:
	d.visualize(draw_boxes=True)
	
print(len(data))
