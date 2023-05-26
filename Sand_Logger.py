#!/usr/bin/env python3

from pathlib import Path
import sys
import cv2
import depthai as dai
import numpy as np
import time
import openpyxl
from openpyxl import Workbook
import xlsxwriter
import os
import datetime
import tkinter as tk
from tkinter import messagebox

# Fetch blob file
nnBlobPath = str((Path(__file__).parent / Path('../Sandman Demo/models/FYP-sandman-e300.blob')).resolve().absolute())

if not Path(nnBlobPath).exists():
    import sys
    raise FileNotFoundError(f'Blob file not found. Check that it is in the /models/ folder')

# YOLO label texts
labelMap = ["bus","car","dog","motorcycle","person","truck"]

# Create pipeline
pipeline = dai.Pipeline()

# Define sources and outputs
camRgb = pipeline.create(dai.node.ColorCamera)
spatialDetectionNetwork = pipeline.create(dai.node.YoloSpatialDetectionNetwork)
monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)
stereo = pipeline.create(dai.node.StereoDepth)
nnNetworkOut = pipeline.create(dai.node.XLinkOut)

xoutRgb = pipeline.create(dai.node.XLinkOut)
xoutNN = pipeline.create(dai.node.XLinkOut)
xoutDepth = pipeline.create(dai.node.XLinkOut)

xoutRgb.setStreamName("rgb")
xoutNN.setStreamName("detections")
xoutDepth.setStreamName("depth")
nnNetworkOut.setStreamName("nnNetwork")

syncNN = True

# Properties
camRgb.setPreviewSize(416, 416)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setInterleaved(False)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
camRgb.setFps(30)

monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

# Setting node configs
stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
# Align depth map to the perspective of RGB camera, on which inference is done
stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
stereo.setOutputSize(monoLeft.getResolutionWidth(), monoLeft.getResolutionHeight())

spatialDetectionNetwork.setBlobPath(nnBlobPath)
spatialDetectionNetwork.setConfidenceThreshold(0.60)
spatialDetectionNetwork.input.setBlocking(False)
spatialDetectionNetwork.setBoundingBoxScaleFactor(0.5)
spatialDetectionNetwork.setDepthLowerThreshold(100)
spatialDetectionNetwork.setDepthUpperThreshold(30000)

# Yolo specific parameters
spatialDetectionNetwork.setNumClasses(6)
spatialDetectionNetwork.setCoordinateSize(4)
spatialDetectionNetwork.setAnchors([10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326])
spatialDetectionNetwork.setAnchorMasks({"side52": [0, 1, 2], "side26": [3, 4, 5], "side13": [6, 7, 8]})
spatialDetectionNetwork.setIouThreshold(0.5)

# Linking
monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)

camRgb.preview.link(spatialDetectionNetwork.input)
if syncNN:
    spatialDetectionNetwork.passthrough.link(xoutRgb.input)
else:
    camRgb.preview.link(xoutRgb.input)

spatialDetectionNetwork.out.link(xoutNN.input)

stereo.depth.link(spatialDetectionNetwork.inputDepth)
spatialDetectionNetwork.passthroughDepth.link(xoutDepth.input)
spatialDetectionNetwork.outNetwork.link(nnNetworkOut.input)

# Defining functions for data logging and message box
def log_data(data_to_log,img_name, bbox, folder_img_dir):
    folder_img_dir = os.path.join(folder_img_dir)
    global log_flag
    global sheet
    if not os.path.exists(folder_img_dir):
        os.makedirs(folder_img_dir)
    if log_flag:
        sheet.append(data_to_log)
    if bbox.size > 0:
            cv2.imwrite(img_name, bbox)

def check_distance(distance, ranges, messages):
    for i, (lower, upper) in enumerate(ranges):
        global msg_flag
        if lower <= distance <= upper and msg_flag:
            messagebox.showinfo("Object Detected", messages[i])


# Range and Message for which message box will call
ranges = [(0,3), (3.0001, 5)]
messages = ["Warning! The object is within the range of 0 to 3 meters!",
            "The object is within the range of 3 to 5 meters!"]

# Data log toggle
log_flag = True
# Message box toggle
msg_flag = False
# Folder name for image directory
folder_img_dir = 'images'
sheet_path = 'datalogger detections.xlsx'

# Workbook Initialization
book = Workbook()
sheet = book.active
sheet.append(("Class", "Date and Time", "Confidence Level", "Distance", "File Name" ,"Image Link", "Warnings")) # Sheet labels

# Connect to device and start pipeline
with dai.Device(pipeline) as device:

    # Output queues will be used to get the rgb frames and nn data from the outputs defined above
    previewQueue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    detectionNNQueue = device.getOutputQueue(name="detections", maxSize=4, blocking=False)
    depthQueue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
    networkQueue = device.getOutputQueue(name="nnNetwork", maxSize=4, blocking=False)

    startTime = time.monotonic()
    counter = 0
    fps = 0
    color = (255, 255, 255)
    printOutputLayersOnce = True

    while True:
        inPreview = previewQueue.get()
        inDet = detectionNNQueue.get()
        depth = depthQueue.get()
        inNN = networkQueue.get()

        if printOutputLayersOnce:
            toPrint = 'Output layer names:'
            for ten in inNN.getAllLayerNames():
                toPrint = f'{toPrint} {ten},'
            print(toPrint)
            printOutputLayersOnce = False;

        frame = inPreview.getCvFrame()
        depthFrame = depth.getFrame() # depthFrame values are in millimeters

        depthFrameColor = cv2.normalize(depthFrame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
        depthFrameColor = cv2.equalizeHist(depthFrameColor)
        depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_HOT)

        counter+=1
        current_time = time.monotonic()
        frame_with_boxes = frame.copy()
        if (current_time - startTime) > 1 :
            fps = counter / (current_time - startTime)
            counter = 0
            startTime = current_time

        detections = inDet.detections

        # If the frame is available, draw bounding boxes on it and show the frame
        height = frame.shape[0]
        width  = frame.shape[1]

        for detection in detections:
            time_ref = datetime.datetime.strftime(datetime.datetime.now(), "%Y-%m-%d %H:%M:%S.%f")[:-3] # Time Reference
            roiData = detection.boundingBoxMapping
            roi = roiData.roi
            roi = roi.denormalize(depthFrameColor.shape[1], depthFrameColor.shape[0])
            topLeft = roi.topLeft()
            bottomRight = roi.bottomRight()
            xmin = int(topLeft.x)
            ymin = int(topLeft.y)
            xmax = int(bottomRight.x)
            ymax = int(bottomRight.y)
            cv2.rectangle(depthFrameColor, (xmin, ymin), (xmax, ymax), color, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX)

            # Denormalize bounding box
            x1 = int(detection.xmin * width)
            x2 = int(detection.xmax * width)
            y1 = int(detection.ymin * height)
            y2 = int(detection.ymax * height)
            try:
                label = labelMap[detection.label]
            except:
                label = detection.label

            # Class, Confidence, and Distance Labels
            x_dist = detection.spatialCoordinates.x / 1000
            y_dist = detection.spatialCoordinates.y / 1000
            z_dist = detection.spatialCoordinates.z / 1000

            cv2.putText(frame, str(label), (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.putText(frame, f"{int(detection.confidence * 100)} %", (x1 + 10, y1 + 35), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            #cv2.putText(frame, f"{int(detection.spatialCoordinates.z) / 1000} m", (x1 + 10, y1 + 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            #cv2.putText(frame, f"X: {int(detection.spatialCoordinates.x) / 1000} m", (x1 + 10, y1 + 50),cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            #cv2.putText(frame, f"Y: {int(detection.spatialCoordinates.y) / 1000} m", (x1 + 10, y1 + 65),cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            #cv2.putText(frame, f"Z: {int(detection.spatialCoordinates.z) / 1000} m", (x1 + 10, y1 + 85),cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            if detection.spatialCoordinates.z == 0 :
                cv2.putText(frame, f"out of range", (x1 + 10, y1 + 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            else:
                cv2.putText(frame, f"{int(detection.spatialCoordinates.z) / 1000} m", (x1 + 10, y1 + 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)


            # Drawing the bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, cv2.FONT_HERSHEY_SIMPLEX)

            # Message box when objects enter range
            if cv2.waitKey(10) == ord('f'):
                msg_flag = not msg_flag  # toggle logging state
                if msg_flag:
                    warning_msg = messages
                    print("Message box enabled")
                else:
                    warning_msg = ""
                    print("Message box disabled")
            check_distance(z_dist, ranges, messages)

            # Data logging when log flag is set to True
            if log_flag:
                if z_dist < 1:
                  bbox = frame[y1:y2, x1:x2]
                  img_name = os.path.join(folder_img_dir, f"{label}_{time_ref.replace(':', '-')}_{detection.spatialCoordinates.z / 1000}.jpg")
                  append_msg = "Level 2 Warning"
                  data_to_log = label, time_ref, detection.confidence * 100, detection.spatialCoordinates.z / 1000, img_name , f'=Hyperlink("{img_name}","{label}")', append_msg

                  log_data(data_to_log,img_name,bbox,folder_img_dir)

                if z_dist in range(1,2):
                  bbox = frame[y1:y2, x1:x2]
                  img_name = os.path.join(folder_img_dir, f"{label}_{time_ref.replace(':', '-')}_{detection.spatialCoordinates.z / 1000}.jpg")
                  append_msg = "Level 1 Warning"
                  data_to_log = label, time_ref, detection.confidence * 100, detection.spatialCoordinates.z / 1000, img_name , f'=Hyperlink("{img_name}","{label}")', append_msg

                  log_data(data_to_log, img_name, bbox, folder_img_dir)
                if z_dist > 2:
                  bbox = frame[y1:y2, x1:x2]
                  img_name = os.path.join(folder_img_dir, f"{label}_{time_ref.replace(':', '-')}_{detection.spatialCoordinates.z / 1000}.jpg")
                  append_msg = "No Warning"
                  data_to_log = label, time_ref, detection.confidence * 100, detection.spatialCoordinates.z / 1000, img_name, f'=Hyperlink("{img_name}","{label}")' , append_msg

                  log_data(data_to_log, img_name, bbox, folder_img_dir)

            if cv2.waitKey(10) == ord('d'):
                log_flag = not log_flag  # toggle logging state
                if log_flag:
                    print("Data logging enabled")
                else:
                    print("Data logging disabled")



        book.save(sheet_path)

        cv2.putText(frame, "NN fps: {:.2f}".format(fps), (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color)
        cv2.imshow("depth", depthFrameColor)
        cv2.namedWindow("Sandman", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Sandman", 640, 640)
        cv2.imshow("Sandman", frame)

        if cv2.waitKey(1) == ord('q'):
            break