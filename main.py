import cv2
from src.config import Config
from src.camera import Camera
from src.detector import Detector
from src.visualizer import Visualizer

def main():
    
    config = Config()
    detector = Detector(config)
    visualizer = Visualizer()

    with Camera(config.camera_index) as cam:
        
        while True:
            frame = cam.read()
            result = detector.detect(frame)
            output = visualizer.draw(frame, result)

            cv2.imshow("YOLOv8 - Detection", output)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()