from ultralytics import YOLO
model=YOLO('yolov8n.pt')
results = model.train(data='aqua_dataset/data.yaml', epochs=20, imgsz=640,device="cpu")