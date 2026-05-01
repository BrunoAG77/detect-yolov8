import cv2

class Camera:
    def __init__(self, index: int = 0):
        self.cap = cv2.VideoCapture(index)
    
    def read(self):
        ret, frame = self.cap.read()

        if not ret:
            raise RuntimeError("[ERRO] falha ao capturar frame!!!")
        return frame
    
    def release(self):
        self.cap.release()
    
    def __enter__(self): 
        return self

    def __exit__(self, *_): 
        self.release()