from abc import ABC, abstractmethod
from oak_yolo.DepthProcessor import DepthProcessor
import numpy as np

class CameraInterface(ABC):
    @property
    @abstractmethod
    def depthProcessor(self) -> DepthProcessor:
        pass

    @abstractmethod
    def get_HFOV(self) -> np.rad:
        pass

    @abstractmethod
    def start(self) -> None:
        pass

    @abstractmethod
    def get_frame(self) -> dict:
        pass