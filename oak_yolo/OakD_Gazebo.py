from oak_yolo.CameraInterface import CameraInterface
from oak_yolo.DepthProcessor import DepthProcessor
import threading
from gz import transport14
from gz.msgs11.pointcloud_packed_pb2 import PointCloudPacked
import numpy as np
import cv2
import time

class OakD_Gazebo(CameraInterface):
    def __init__(self,width:int,height:int,HFOV:bool):
        self.depth_frame = None
        self.rgb_frame = None
        self.update_thread = None
        self.frame_lock = threading.Lock()
        self.node = transport14.Node()
        self.topic = "depth_camera/points"
        self.width = width
        self.height = height
        self.HFOV = HFOV
        self._depthProcessor = None
        self.__setup()

    @property
    def depthProcessor(self):
        return self._depthProcessor

    def update_func(self):
        self.node.subscribe(msg_type=PointCloudPacked,topic=self.topic,callback=self.on_depth_points)
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("Exiting...")

    def __setup(self):
        self.update_thread = threading.Thread(target=self.update_func)

    def get_HFOV(self):
        return self.HFOV

    def start(self):
        self.update_thread.start()
        self._depthProcessor = DepthProcessor(width=self.width,height=self.height,hfov=self.HFOV)
    
    def on_depth_points(self,msg:PointCloudPacked):
        with self.frame_lock:
            # Convert raw data into a numpy array
            points = np.frombuffer(msg.data, dtype=np.float32)
            width = msg.width
            height = msg.height
            num_points = width * height
            row_step = msg.row_step
            point_step = msg.point_step
            # Given point_step is 24 bytes, each point has 24/4 = 6 floats.
            floats_per_point = int(point_step/4)
            try:
                points = points.reshape((msg.height, msg.width, floats_per_point))
            except Exception as e:
                print("Error reshaping:", e)
                return

            depth_image = np.abs(points[:,:,2])
            min_depth, max_depth = 0, 100  # adjust based on expected range
            depth_clipped = np.clip(depth_image, min_depth, max_depth)

            self.depth_frame = depth_image

            depth_normalized = ((depth_clipped - min_depth) / (max_depth - min_depth) * 255).astype(np.uint8)
            packed_rgb = points[:, :, 4]
            # packed_rgb_uint = packed_rgb.view(np.uint32)
            # r = ((packed_rgb_uint >> 16) & 0xFF).astype(np.uint8)
            # g = ((packed_rgb_uint >> 8) & 0xFF).astype(np.uint8)
            # b = (packed_rgb_uint & 0xFF).astype(np.uint8)
            # rgb_image = np.stack((r, g, b), axis=-1)  # Shape: (height, width, 3).
            # bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
            # cv2.imshow("RGB Image", bgr_image)

    def get_frame(self):
        with self.frame_lock:
            return {"depth":self.depth_frame,"rgb":self.rgb_frame}