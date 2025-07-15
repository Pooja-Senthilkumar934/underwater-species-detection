from ultralytics import YOLO
# from ultralytics import show_result_pyplot
model=YOLO('yolov8n.pt')
model=YOLO('./runs/detect/train16/weights/best.pt')
# Validate the model
# metrics = model.val()  # no arguments needed, dataset and settings remembered
# metrics.box.map    # map50-95
# metrics.box.map50  # map50
# metrics.box.map75  # map75
# metrics.box.maps   # a list contains map50-95 of each category

# Predict with the model
results = model('./new/comrawpixel3285239.jpg')
# results.show()
# show_result_pyplot(results)
for r in results:
    print(r.boxes)
    