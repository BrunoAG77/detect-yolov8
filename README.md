# Detecção de objetos com YOLOv8

Este projeto implementa um sistema de detecção de objetos em tempo real utilizando visão computacional e o modelo YOLOv8 (You Only Look Once).
A aplicação captura frames da webcam, processa cada imagem com um modelo de deep learning e exibe os objetos detectados com bounding boxes e labels diretamente na tela. O foco do projeto é demonstrar na prática a aplicação de inteligência artificial em visão computacional, com foco em desempenho e organização de código.

## Principais Funcionalidades
- Detecção de objetos em tempo real via webcam
- Arquitetura modular (Camera, Detector, Visualizer, Config)
- Uso do YOLOv8 com a biblioteca Ultralytics
- Configuração simples e adaptável (confiança, device, classes)
- Interface leve com OpenCV

## Fluxo do sistema
Captura de imagem da câmera -> Processamento com o modelo YOLOv8 -> Identificação de objetos e suas localizações -> Renderização dos resultados na tela

## Estrutura do projeto 
```
detect-yolov8/
├── models/
│   ├── yolov8n.pr
├── src/
│   ├── __pycache__/
│   ├── ├── __init__.cpython-314.pyc
│   ├── ├── camera.cpython-314.pyc
│   ├── ├── config.cpython-314.pyc
│   ├── ├── detector.cpython-314.pyc
│   ├── ├── visualizer.cpython-314.pyc
│   ├── __init__.py
│   ├── camera.py
│   ├── config.py
│   ├── detector.py
│   ├── visualizer.py
├── README.md
├── main.py
```

## Arquitetura modular
### Camera
A câmera é responsável por acessar a webcam e capturar frames, usando `cv2.VideoCapture()`. O erro já é tratado se não conseguir ler a imagem.
```
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
```

### Detector
No detector, o YOLO já consegue detectar tudo sendo carregado uma única vez, rodando a predição frame por frame. O detector também configura o nível de confiança e o dispositivo (CPU e GPU), retornando somente o primeiro resultado.
```
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
```

### Visualizer
O visualizador é responsável por desenhar/plotar os resultados, isto é, a caixa de detecção junto do YOLO, e adicionando o texto na tela graças ao OpenCV, por fim, retornando o frame já anotado.
```
import cv2

class Visualizer:
    
    def draw(self, frame, result):
        annotated = result.plot()
        
        fps_text = "Pressione Q para sair"

        cv2.putText(annotated, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        
        return annotated
```

### Config
Centralização de todas as configurações, incluindo o caminho do modelo, nível confiança, dispositivo (CPU e GPU), câmera e as classes de objetos.
```
from dataclasses import dataclass, field

@dataclass
class Config:
    model_path: str = "models/yolov8n.pt"
    confidence: float = 0.5
    device: str = "cpu"         
    camera_index: int = 0
    target_classes: list = field(default_factory=lambda: ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"])
```

### Main
Arquivo principal do programa; inicializando Config, Detector e Visualizer; após a inicialização um loop de captura -> detecção -> renderização é executado,mostrando o resultado com `cv2.imshow()`. A tecla `Q` quebra o loop e encerra a aplicação.
```
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
```

## Tecnologias utilizadas
- Python
- OpenCV
- Ultralytics YOLOv8
- Ambiente Virtual/Virtual Environment (venv)

## Tutorial
A execução deve ser feita diretamente pelo terminal.

### Clonagem do repositório
```git clone https://github.com/BrunoAG77/detect-yolov8```

```cd detect-yolov8```

### Criação do ambiente virtual
```python -m venv venv```

### Ativação do ambiente virtual
Windows: ```venv\Scripts\activate```

Linux/Mac: ```source venv/bin/activate```

### Instalação das bibliotecas, se necessário
```pip install opencv-python```

```pip install ultralytics```

### Execução do projeto
```python main.py```

### Encerramento do projeto
Pressione "Q" para encerrar a aplicação.

## Referencias: 
https://www.youtube.com/watch?v=O9Jbdy5xOow

https://dipankarmedh1.medium.com/real-time-object-detection-with-yolo-and-webcam-enhancing-your-computer-vision-skills-861b97c78993

https://github.com/codershiyar/object-detection-using-webcam

https://www.youtube.com/watch?v=YKbBXWBJloY

https://docs.ultralytics.com/modes/predict/#real-world-applications
