import cv2
import numpy as np
def get_face_detector(modelFile=None, configFile=None, quantized=None):
    if quantized:
        if modelFile == None:
            modelFile = "models/opencv_face_detector_vint8.pb"
        if configFile == None:
            comnfigFile = "models/opencv_face_detector.pbtxt"
        model = cv2.dnn.readNetFromTensorflow(modelFile, configFile)
    else:
        if modelFile== None:
            modelFile = "models/res10_300x300_ssd_iter_140000.caffemodel"
        if configFile == None:
            configFile = "models/deploy.prototxt"
        model = cv2.dnn.readNetFromCaffe(configFile, modelFile)
    return model
    
        