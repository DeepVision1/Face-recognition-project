from ultralytics import YOLO

model = YOLO("Models/Eye_Model.pt")
results = model.predict(source=0, show=True, conf=0.5)  #save=True to save video