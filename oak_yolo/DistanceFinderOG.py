import cv2
import depthai as dai
import numpy as np
from ultralytics import YOLO
import time
from datetime import timedelta
import torch
fps = 30

# Create pipeline
pipeline = dai.Pipeline()

# Define sources and outputs
rgb = pipeline.create(dai.node.Camera)
monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)
stereo = pipeline.create(dai.node.StereoDepth)

sync = pipeline.create(dai.node.Sync)
sync.setSyncThreshold(timedelta(milliseconds=50))
spatialLocationCalculator = pipeline.create(dai.node.SpatialLocationCalculator)

xoutRGB = pipeline.create(dai.node.XLinkOut)
xoutDepth = pipeline.create(dai.node.XLinkOut)
xoutSpatialData = pipeline.create(dai.node.XLinkOut)
xinSpatialCalcConfig = pipeline.create(dai.node.XLinkIn)

syncOut = pipeline.create(dai.node.XLinkOut)
syncOut.setStreamName("syncOut")

xoutRGB.setStreamName("rgb")
xoutDepth.setStreamName("depth")
xoutSpatialData.setStreamName("spatialData")
xinSpatialCalcConfig.setStreamName("spatialCalcConfig")

# Camera Properties
rgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
rgb.setFps(fps)
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoLeft.setFps(fps)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)
monoRight.setFps(fps)

# Stereo Depth Config
stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
stereo.setLeftRightCheck(True)
stereo.setSubpixel(True)
stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)

# Spatial Location Calculator Config
spatialLocationCalculator.inputConfig.setWaitForMessage(False)
topLeft = dai.Point2f(0.4, 0.4)
bottomRight = dai.Point2f(0.6, 0.6)

config = dai.SpatialLocationCalculatorConfigData()
config.depthThresholds.lowerThreshold = 100
config.depthThresholds.upperThreshold = 10000
config.calculationAlgorithm = dai.SpatialLocationCalculatorAlgorithm.MEDIAN

config.roi = dai.Rect(topLeft, bottomRight)
spatialLocationCalculator.initialConfig.addROI(config)

# Linking
#rgb.preview.link(xoutRGB.input)
rgb.preview.link(sync.inputs["rgbSync"])
rgb.setPreviewSize(1280,720)
monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)
stereo.depth.link(spatialLocationCalculator.inputDepth)
#spatialLocationCalculator.passthroughDepth.link(xoutDepth.input)
spatialLocationCalculator.passthroughDepth.link(sync.inputs["depthSync"])
spatialLocationCalculator.out.link(xoutSpatialData.input)
xinSpatialCalcConfig.out.link(spatialLocationCalculator.inputConfig)

sync.out.link(syncOut.input)
# YOLO Model
#model = YOLO("yolo11n.pt")
model = YOLO("../yolo11n.engine",task="detect")
# Connect to device and start pipeline
with dai.Device(pipeline,maxUsbSpeed=dai.UsbSpeed.SUPER) as device:
    device.setLogOutputLevel(dai.LogLevel.TRACE)
    device.setIrFloodLightIntensity(1)
    device.setIrLaserDotProjectorIntensity(1)
    synced = device.getOutputQueue(name="syncOut", maxSize=1,blocking=False)
    #depthQueue = device.getOutputQueue(name="depth", maxSize=1, blocking=False)
    spatialCalcQueue = device.getOutputQueue(name="spatialData", maxSize=1, blocking=False)
    
    spatialCalcConfigInQueue = device.getInputQueue("spatialCalcConfig")
    #rgbQueue = device.getOutputQueue(name="rgb")
    i = 0
    while True:
        start_time = time.time()
        section_start = time.time()
        #synced testing
        syncData = synced.get()
        inDepth = None
        inRGB = None

        for name,msg in syncData:
            # print(name,msg)
            # cv2.imshow(name,msg.getCvFrame())
            if name == "depthSync":
                inDepth = msg
            if name == "rgbSync":
                inRGB = msg

        # Get the frame from the camera
        depthFrame = inDepth.getFrame()

        rgbFrame = inRGB.getCvFrame()
        print(f'Got Frames: {time.time()-section_start}')
        section_start=time.time()
        # Colorize the depth frame
        depthFrameColorized = cv2.applyColorMap(cv2.convertScaleAbs(depthFrame, alpha=0.03), cv2.COLORMAP_JET)

        # Depth processing and Dynamic ROI based on YOLO detection
        frame = depthFrameColorized 
        results = model(rgbFrame,device=torch.device("cuda"))
        print(f'Model Predictions: {time.time()-section_start}')
        section_start=time.time()

       	if len(results[0].boxes) == 0:
            combined = np.concatenate([frame,rgbFrame],axis=0)
            # cv2.imshow("Depth", frame)
            # cv2.imshow("RGB",rgbFrame)
            end_time = time.time()
            frame_time = end_time - start_time
            print(f'Frame Time: {frame_time}')
            cv2.putText(combined,str(1/frame_time),(30,40),cv2.FONT_HERSHEY_TRIPLEX, 2 , (0,0,255))
            cv2.imshow("Combined",combined)
            print(f'Show no detections {combined.shape}')
            if cv2.waitKey(1) == ord('q'):
            	break
            continue
       
        

        # Iterate through YOLO detections and send updated ROI
        for detection in results:
            roi_list=[]
            for box in detection.boxes:
                x1, y1, x2, y2= box.xyxy[0]
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                cls = box.cls.item()
                cls = model.names[cls]
                #print(f"Detected object at ({x1}, {y1}) and ({x2}, {y2})")
                
                # Define the ROI for spatial calculation
                topLeft = dai.Point2f(x1 / frame.shape[1], y1 / frame.shape[0])
                bottomRight = dai.Point2f(x2 / frame.shape[1], y2 / frame.shape[0])
                
                #roi_list.append(dai.Rect(topLeft, bottomRight))
                config = dai.SpatialLocationCalculatorConfigData()
                config.depthThresholds.lowerThreshold = 100
                config.depthThresholds.upperThreshold = 10000
                config.calculationAlgorithm = dai.SpatialLocationCalculatorAlgorithm.MEDIAN
                config.roi = dai.Rect(topLeft, bottomRight)
                roi_list.append(config)
                cv2.putText(rgbFrame,cls,(x1+10,y1+20),cv2.FONT_HERSHEY_TRIPLEX, 0.5 , (0,0,255))
        print(f'Add ROIs: {time.time()-section_start}')
        section_start=time.time()
        
        cfg = dai.SpatialLocationCalculatorConfig()
        cfg.setROIs(roi_list)
        spatialCalcConfigInQueue.send(cfg)
        
        spacialData = spatialCalcQueue.get().getSpatialLocations()
        #print(len(spacialData))
        for object in spacialData:
            roi = object.config.roi
            roi = roi.denormalize(width=1280, height=720)
            xmin = int(roi.topLeft().x)
            ymin = int(roi.topLeft().y)
            xmax = int(roi.bottomRight().x)
            ymax = int(roi.bottomRight().y)

            depthMin = object.depthMin
            depthMax = object.depthMax
            color=(0,0,255)
            fontType = cv2.FONT_HERSHEY_TRIPLEX
            cv2.rectangle(rgbFrame, (xmin, ymin), (xmax, ymax), color, 1)
            cv2.putText(rgbFrame, f"X: {int(object.spatialCoordinates.x)} mm", (xmin + 10, ymin + 35), fontType, 0.5, color)
            cv2.putText(rgbFrame, f"Y: {int(object.spatialCoordinates.y)} mm", (xmin + 10, ymin + 50), fontType, 0.5, color)
            cv2.putText(rgbFrame, f"Z: {int(object.spatialCoordinates.z)} mm", (xmin + 10, ymin + 65), fontType, 0.5, color)
        print(f'Plot Depth Data: {time.time()-section_start}')
        combined = np.concatenate([frame,rgbFrame],axis=0)
        # cv2.imshow("Depth", frame)
        # cv2.imshow("RGB",rgbFrame)
        end_time = time.time()
        frame_time = end_time - start_time
        print(f'Frame Time: {frame_time}')
        cv2.putText(combined,str(1/frame_time),(30,40),cv2.FONT_HERSHEY_TRIPLEX, 2 , (0,0,255))
        cv2.imshow("Combined",combined)

        if cv2.waitKey(1) == ord('q'):
            break

cv2.destroyAllWindows()
