from ultralytics import YOLO
from .config import Config

class Detector:
    
    def __init__(self, config: Config):
        self.model = YOLO(config.model_path)
        self.config = config

    def detect(self, frame):
    
        results = self.model.predict(
            frame,
            conf=self.config.confidence,
            device=self.config.device,
            verbose=False
        )
    
        return results[0]