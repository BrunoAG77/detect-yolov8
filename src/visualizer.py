import cv2

class Visualizer:
    
    def draw(self, frame, result):
        annotated = result.plot()
        
        fps_text = "Pressione Q para sair"

        cv2.putText(annotated, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        
        return annotated