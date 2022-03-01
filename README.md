# Yolov5 Real Time Object Detection model 
![framework](https://img.shields.io/badge/framework-flask-red)
![libraries](https://img.shields.io/badge/libraries-opencv-green)
![models](https://img.shields.io/badge/models-yolov5-yellow)

The Yolov5s pretained model is deployed using flask.
This repo contains example apps for exposing the [yolo5](https://github.com/ultralytics/yolov5) object detection model from [pytorch hub](https://pytorch.org/hub/ultralytics_yolov5/) via a [flask](https://flask.palletsprojects.com/en/1.1.x/) api/app.



## Web app
Simple app that enables live webcam detection using pretrained YOLOv5s weights and see real time inference result of the model in the browser.

![yolov5-real-time](https://user-images.githubusercontent.com/69728128/156182901-98c58df9-d23f-4e92-a4aa-7a9d9dc8ba67.JPG)

## Run & Develop locally
Run locally and dev:
* `conda create -n <VENV>`
* `conda activate <VENV>`
* `(<VENV>) $ pip install -r requirements.txt`
* `(<VENV>) $ flask run`

## Docker
The example dockerfile shows how to expose the rest API:
```
# Build
docker build -t yolov5 .
# Run
docker run -p 5000:5000 yolov5-flask:latest
```

## reference
- https://github.com/ultralytics/yolov5
- https://github.com/jzhang533/yolov5-flask
- https://github.com/avinassh/pytorch-flask-api-heroku
