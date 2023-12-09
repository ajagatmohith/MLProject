import os
import gc
import cv2
import copy
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from mtcnn.mtcnn import MTCNN
from itertools import product
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, accuracy_score

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
import datetime
import tensorflow as tf

from keras.models import Sequential, load_model
from keras.utils import to_categorical
from keras.optimizers import RMSprop

from keras.layers import (
    Conv2D,
    BatchNormalization,
    MaxPooling2D,
    Dropout,
    Flatten,
    Activation,
    Dense,
)
from keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard,
)
from keras.preprocessing.image import ImageDataGenerator
physicalDevices = tf.config.list_physical_devices("GPU")
print(physicalDevices)
if len(physicalDevices) > 0:
    tf.config.experimental.set_memory_growth(physicalDevices[0], True)


trainDataFolderPath = "Data/train/"
testDataFolderPath = "Data/test/"

def DrawBoundaryBoxs(
    frame: np.ndarray,
    boundryBox: list,
    color: tuple = (0, 255, 0),
    thickness: int = 2,
):
    [x, y, w, h] = boundryBox
    frame = cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
    return frame

def DisplayPrediction(frame: np.ndarray, Name: str = None):
    if(not Name):
        Name =  "No Face Detected"
    frame_width = frame.shape[1]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (0, 255, 0)  # Green color
    line_type = 2
    text_size = cv2.getTextSize(Name, font, font_scale, line_type)[0]
    text_x = (frame_width - text_size[0]) // 2  # Centered horizontally
    text_y = 30 
    frame = cv2.putText(frame, Name, (text_x, text_y), font, font_scale, font_color, line_type)
    return frame
def DetectFaces(frame: np.ndarray, faceCascade: MTCNN):
    # grayScaleImage = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detectedFacesBBoxs = faceCascade.detect_faces(frame)
    
    detectedFacesBBoxs = [detectedFacesBBox['box'] for detectedFacesBBox in detectedFacesBBoxs]
    return detectedFacesBBoxs

def CropImage(image: np.ndarray, bBox: list):
    [x, y, w, h] = bBox
    croppedImage = image[y : y + h, x : x + w]
    return croppedImage

def ResizeImage(image: np.ndarray, resize: tuple = (100, 100)):
    image = cv2.resize(image, resize)
    return image

def LoadDataSet(folderPath: str):
    if not (os.path.exists(folderPath)):
        print("Please ensure that FolderPath is valid")
        return None
    labelNames = os.listdir(folderPath)
    imageDataset = []
    labelDataset = []
    for folderName in labelNames:
        if not folderName == ".DS_Store":
            imageList = os.listdir(folderPath + folderName)
            for imageName in tqdm(imageList):
                if not imageName == ".DS_Store":
                    imageDataset.append(
                        # ResizeImage(
                            cv2.imread(folderPath + folderName + "/" + imageName)
                        # )
                    )
                    labelDataset.append(folderName)

    return imageDataset, labelDataset

def GetMaxAreabBox(bBoxes: list, imageArea: int):
    bestbBox = []
    maxArea = 0
    for bBox in bBoxes:
        [x, y, w, h] = bBox
        if w * h > maxArea:
            maxArea = w * h
            bestbBox = bBox
    return bestbBox

def TransformFaces(imageDataset: list, labelDataset: list, min_face_size: int = 20):
    faceCascade = MTCNN(min_face_size=min_face_size)
    faceImages = []
    labels = []
    zipped = zip(imageDataset, labelDataset)
    for image, label in tqdm(zipped):
        bBoxes = DetectFaces(image, faceCascade)
        imageArea = image.shape[0] * image.shape[1]
        
        if len(bBoxes):
            bBox = GetMaxAreabBox(bBoxes, imageArea)
            if len(bBox):
                faceImage = CropImage(image, bBox)
                faceImages.append(faceImage)
                labels.append(label)
    return faceImages, labels

trainingImageDataset, trainingLabelDataset = LoadDataSet(trainDataFolderPath)
encoder = LabelEncoder()
encoder = encoder.fit(trainingLabelDataset)
trainingLabelDataset = encoder.transform(trainingLabelDataset)
trainingLabelDataset = to_categorical(trainingLabelDataset, num_classes=4)


def open_webcam(model: Sequential):
    AllPrediction = []
    faceCascade = MTCNN(min_face_size=20)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        bBoxes = DetectFaces(frame, faceCascade)
        imageArea = frame.shape[0]*frame.shape[1]
        if len(bBoxes):
            bBox = GetMaxAreabBox(bBoxes, imageArea)
            bBoxImage = DrawBoundaryBoxs(frame, bBox)
            croppedImage = CropImage(frame, bBox)
            croppedImage = ResizeImage(croppedImage)
            prediction = model.predict(croppedImage.reshape(1, 100, 100, 3))
            prediction = encoder.inverse_transform([np.argmax(prediction)])
            if(prediction[0] not in AllPrediction):
                AllPrediction.append(prediction[0])

            bBoxImage = DisplayPrediction(bBoxImage, prediction[0])
        else:
            bBoxImage = DisplayPrediction(frame)
        cv2.imshow("Testing the image", bBoxImage)

            
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    AllPrediction = pd.DataFrame(AllPrediction, columns=['Students Attended'])
    AllPrediction.to_csv("Predictions.csv", sep=",")
        
    
model = load_model('MyModel.h5')

open_webcam(model)

