from oak_yolo.CameraInterface import CameraInterface
from oak_yolo.DepthProcessor import DepthProcessor
import depthai as dai
from datetime import timedelta
import numpy as np

class OakD_Cam(CameraInterface):
    def __init__(self, camera_settings: dict):
        """
        Constructor for OakD_Cam object
        """
        self._depthProcessor = None
        self.camera_settings = camera_settings
        # OakD Device Camera
        self.device: dai.Device = None
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

        self.__camera_setup()

    @property
    def depthProcessor(self):
        return self._depthProcessor

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

    def get_HFOV(self):
        if self.device:
            calibData = self.device.readCalibration()
            return np.deg2rad(calibData.getFov(dai.CameraBoardSocket.CAM_A,useSpec=False))

    def start(self):
        # Connect To Device And Start Pipeline
        self.device = dai.Device(self.pipeline,maxUsbSpeed=dai.UsbSpeed.SUPER)
        # Add Device Settings
        self.device.setIrFloodLightIntensity(self.camera_settings["floodLightIntensity"])
        self.device.setIrLaserDotProjectorIntensity(self.camera_settings["laserDotProjectorIntensity"])
        self.synced = self.device.getOutputQueue(name="sync",maxSize=1,blocking=False)
        self._depthProcessor = DepthProcessor(width=self.camera_settings["previewRes"][0],
                                              height=self.camera_settings["previewRes"][1],
                                              hfov=self.get_HFOV())

    def get_frame(self):
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
        return {"depth":depthFrame,"rgb":rgbFrame}
    

