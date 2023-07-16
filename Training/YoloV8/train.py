# https://github.com/computervisioneng/image-segmentation-yolov8

from ultralytics import YOLO  # to load and use YOLO V8
import os

model = YOLO('yolov8n-seg.pt')  # load a pretrained model (recommended for training)


current_directory = os.getcwd()
print(current_directory)


#script_directory = os.path.dirname(os.path.abspath(__file__))

script_directory = 'C:\\Users\\dezos\\Documents\\Fibres\\FibreAnalysis\\Training\YoloV8\\'

config_path = os.path.join(script_directory, 'config.yaml')
print(config_path)
model.train(data=config_path, epochs=1, imgsz=640)


# example for testing etc at https://docs.ultralytics.com/usage/python/