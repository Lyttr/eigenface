#! python3
 # -*- encoding: utf-8 -*-
'''
@File    :   eigrnface.py
@Time    :   2023/11/13 12:28:41
@Author  :   铲子君 
@Version :   1.0
@Contact :   
'''

import cv2
import numpy as np
import time
import os
import copy
import matplotlib.pyplot as plt
dnnnet = cv2.dnn.readNetFromTensorflow(
    "./opencv_face_detector_uint8.pb", "./opencv_face_detector.pbtxt")
size = (200, 200)


def faceDetector(img):
    """
    @description  :
    use pre trained dnn face detector fron opencv
    @param  :
    img: image to be dtcted
    @Returns  :
    faces: list,coordinations of faces
    """

    h, w = img.shape[:2]
    blobs = cv2.dnn.blobFromImage(img, 1.0, (300, 300), False, False)
    dnnnet.setInput(blobs)
    detections = dnnnet.forward()
    faces = []
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.7:

            detections[0, 0, i, 3:7] = detections[0,
                                                  0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = detections[0, 0, i, 3:7].astype("int")
            faces.append([x1, y1, x2, y2])
            y = y1 - 10 if y1 - 10 > 10 else y1 + 10
            text = "%.3f" % (confidence * 100)+'%'
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(img, text, (x1, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.imshow('faces', img)
    return faces


def saveFace(img, label, id):
    """
    @description  :
    save images
    path: ./label/id.pgm
    @param  :
    img: images to be saved,gray image
    label: name of directory
    id: name of files
    @Returns  :
    none
    """
    
    
    if not os.path.exists(label):
        os.mkdir(label)

    cv2.imwrite(label+'/'+str(id)+'.pgm', img)


def load_dataset(datasetPath):
    """
    @description  :
    load dataset from name1/1.pmg
                           /2.pmg...
                      name2/
                      ... 
    @param  :
    datasetpathL path of dataset
    @Returns  :
    X: numpy array of images
    y: numpy array of id of images
    name: name of images
    """
    
    
    names = []
    X = []
    y = []
    ID = 0
    for name in os.listdir(datasetPath):
        subpath = os.path.join(datasetPath, name)
        if os.path.isdir(subpath):
            names.append(name)
            for file in os.listdir(subpath):
                im = cv2.imread(os.path.join(subpath, file),
                                cv2.IMREAD_GRAYSCALE)
                X.append(np.asarray(im, dtype=np.uint8))
                y.append(ID)
            ID += 1
    X = np.asarray(X)
    y = np.asarray(y, dtype=np.int32)
    return X, y, names


def createDataset():
    """
    @description  :
    creat a dataset ,save it in dataset.npz
    input names of faces
    press A to save a face
    press B to add a new name
    press ESC to exit
    @param  :
    none
    @Returns  :
    none
    """
    
    
    # 捕获内部摄像头
    cap = cv2.VideoCapture(0)
    id = 0
    labelid = 0
    name = np.array(input('Input the new name\n'))
    while(1):
        ret, frame = cap.read()

        frame = cv2.flip(frame, 1)
        img = copy.deepcopy(frame)
        faces = faceDetector(frame)
        key = cv2.waitKey(1)
        if key == 27:
            break
        elif key == 97:

            face = img[faces[0][1]:faces[0][3], faces[0][0]:faces[0][2]]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
            face = cv2.resize(face, size, interpolation=cv2.INTER_LANCZOS4)
            # saveFace(face,label+str(labelid),id)
            if(id == 0):
                X = [np.asarray(face, dtype=np.uint8)]
                y = np.array([0], dtype=np.uint32)
                names = np.array(name)
            else:
                y = np.append(y, [id])
                X = np.append(X, [np.asarray(face, dtype=np.uint8)], axis=0)
                names = np.append(names, name)
            cv2.imshow(np.array2string(name), face)

            id = id+1
        elif key == 98:
            name = np.array(input('Input the new name\n'))

        # cv2.imshow('camera',frame)
    np.savez('dataset', X=X, y=y, names=names)
    cap.release()
    cv2.destroyAllWindows()


def modelEnhance(model, names, X, label, threshold=5000):  
    """
    @description  :
    enhance the model
    *** always keep only the face to be enhanced in the camera
    input the name to be enhanced
    press A to show current face
    press B to input a new name
    @param  :
    model: eigenface model to be enhanced
    names: names of dataset
    X: images in the dataset
    label: id of dataset
    threshold: faces higher than threshold will be added in the dataset and train the model
    @Returns  :
    none
    """
    
    
    cap = cv2.VideoCapture(0)
    id = 0
    name = np.array(input('input the name\n'))
    sum = 0
    id = 0
    while(1):
        ret, frame = cap.read()

        frame = cv2.flip(frame, 1)
        img = copy.deepcopy(frame)
        faces = faceDetector(frame)
        key = cv2.waitKey(1)

        for i in faces:
            try:
                face = img[i[1]:i[3], i[0]:i[2]]
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
                # face=np.clip(face*(np.mean(X)/128),0,255).astype(np.uint8)
                face = cv2.resize(face, size, interpolation=cv2.INTER_LANCZOS4)

                ID_predict, confidence = model.predict(face)
                id = id+1
                sum = sum+confidence

                name = names[ID_predict]
                y = i[1] - 10 if i[1] - 10 > 10 else i[1] + 10
                text = "%d" % (confidence)+name
                cv2.putText(
                    img, text, (i[0], y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                print("name:%s, confidence:%.2f" % (name, confidence))
                cv2.rectangle(img, (i[0], i[1]), (i[2], i[3]), (255, 0, 0), 2)
                if confidence > threshold:

                    X = np.append(
                        X, [np.asarray(face, dtype=np.uint8)], axis=0)
                    #X = np.asarray(X)
                    label = np.append(label, [label[-1]+1])
                    names = np.append(names, name)
                    model.train(X, label)
                    print(X.shape)
            except Exception as e:
                print(e)
        cv2.imshow('enhance', img)
        if key == 27:
            break
        elif key == 97:
            try:
                cv2.imshow('face1', face)
            except:
                pass

        elif key == 98:
            name = np.array(input('input the name\n'))
            sum = 0
            id = 0
        # cv2.imshow('camera',frame)
    np.savez('dataset', X=X, y=label, names=names)
    cap.release()
    cv2.destroyAllWindows()


def modelTest(model, name, threshold):
    """
    @description  :
    test model
    press ESC to exit
    @param  :
    model: model to be tested
    nameL names in dataset
    threshold: faces higher than threshold will be labeled as unknown
    @Returns  :
    none
    """
    
    
    cap = cv2.VideoCapture(0)
    id = 0

    while(1):
        ret, frame = cap.read()

        frame = cv2.flip(frame, 1)
        img = copy.deepcopy(frame)
        faces = faceDetector(frame)
        key = cv2.waitKey(1)

        for i in faces:
            try:
                face = img[i[1]:i[3], i[0]:i[2]]
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
                # face=np.clip(face*(np.mean(X)/128),0,255).astype(np.uint8)
                face = cv2.resize(face, size, interpolation=cv2.INTER_LANCZOS4)

                ID_predict, confidence = model.predict(face)
                y = i[1] - 10 if i[1] - 10 > 10 else i[1] + 10
                if confidence < threshold:

                    name = names[ID_predict]
                    text = "%d" % (confidence)+'   '+name
                    cv2.putText(
                        img, text, (i[0], y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    cv2.rectangle(img, (i[0], i[1]),
                                  (i[2], i[3]), (255, 0, 0), 2)
                else:
                    name = 'unknown '
                    text = "%d" % (confidence)+'   '+name
                    cv2.putText(
                        img, text, (i[0], y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    cv2.rectangle(img, (i[0], i[1]),
                                  (i[2], i[3]), (255, 0, 0), 2)
                print("name:%s, confidence:%.2f" % (name, confidence))

            except Exception as e:
                print(e)
        cv2.imshow('classification', img)
        if key == 27:
            break
        elif key == 97:
            try:
                cv2.imshow('face1', face)
            except:
                pass

        # cv2.imshow('camera',frame)

    cap.release()
    cv2.destroyAllWindows()


# createDataset()
try:
    dataset = np.load('dataset.npz')
except:
    print('no dataset,create dataset\n')
    createDataset()
    dataset = np.load('dataset.npz')
X = dataset['X']
y = dataset['y']
names = dataset['names']
print(X.dtype, X.shape)
intensity = np.mean(X)
model = cv2.face.EigenFaceRecognizer_create()
model.train(X, y)
mean = model.getMean()
# cv2.imshow('meanface',mean.reshape(100,100).astype(np.uint8))


modelEnhance(model, names, X, y, 1500)
model.save('model')
dataset = np.load('dataset.npz')
X = dataset['X']
y = dataset['y']
names = dataset['names']
model.train(X, y)
modelTest(model, names, 4000)
