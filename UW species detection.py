import cv2
from PIL import Image
import torch
from torchvision import transforms
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.plots import plot_one_box

# Load the YOLOv5 model
model = attempt_load(weights='yolov5s.pt', map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

# Define a function to perform object detection
def detect_objects(image_path):
    img = Image.open(image_path)
    img_tensor = transforms.ToTensor()(img)
    img_tensor, _ = model.preprocess(img_tensor, {})
    pred = model(img_tensor.unsqueeze(0))
    pred = non_max_suppression(pred, conf_thres=0.4, iou_thres=0.5)[0]

    if pred is not None:
        pred[:, :4] = scale_coords(img.shape[1:], pred[:, :4], img.size)
    return pred

# Define a function to draw bounding boxes on the image
def draw_boxes(image_path, predictions):
    img = cv2.imread(image_path)
    for x1, y1, x2, y2, conf, cls in predictions:
        plot_one_box((x1, y1, x2, y2), img, label=int(cls), color=(0, 255, 0))
    cv2.imshow('Image with Bounding Boxes', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Test image path
image_path = 'test.jpg'

# Perform object detection
predictions = detect_objects(image_path)

# Draw bounding boxes on the image
draw_boxes(image_path, predictions)
