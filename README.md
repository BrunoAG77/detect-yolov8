 YOLOv8 Real-Time Object Detection
Este projeto implementa um sistema de detecção de objetos em tempo real utilizando visão computacional e o modelo YOLOv8 (You Only Look Once).
A aplicação captura frames da webcam, processa cada imagem com um modelo de deep learning e exibe os objetos detectados com bounding boxes e labels diretamente na tela.

Principais Features: 
- Detecção de objetos em tempo real via webcam
- Arquitetura modular (Camera, Detector, Visualizer, Config)
- Uso do YOLOv8 com a biblioteca Ultralytics
- Configuração simples e adaptável (confiança, device, classes)
- Interface leve com OpenCV

Fluxo do sistema:
1) Captura de imagem da câmera
2) Processamento com o modelo YOLOv8
3) Identificação de objetos e suas localizações
4) Renderização dos resultados na tela

Tecnologias utilizadas:
- Python
- OpenCV
- Ultralytics YOLOv8
- Virtual Environment (venv)

Exemplo de uso:
- Execute o projeto e visualize detecções em tempo real diretamente pela webcam.
- Pressione "Q" para encerrar a aplicação.

Objetivo: Demonstrar na prática a aplicação de inteligência artificial em visão computacional, com foco em desempenho e organização de código.

A execução deve ser feita diretamente pelo terminal.

Referencias: 
https://www.youtube.com/watch?v=O9Jbdy5xOow

https://dipankarmedh1.medium.com/real-time-object-detection-with-yolo-and-webcam-enhancing-your-computer-vision-skills-861b97c78993

https://github.com/codershiyar/object-detection-using-webcam

https://www.youtube.com/watch?v=YKbBXWBJloY

https://docs.ultralytics.com/modes/predict/#real-world-applications
